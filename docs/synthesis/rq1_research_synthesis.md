# RQ1 Research Synthesis — Do Modern LLM-Based Evaluators Reliably Measure Whether a Multiple-Choice Question Genuinely Requires the Target Learning Material?

**Research question.** *Do modern LLM-based evaluators reliably measure whether a multiple-choice question genuinely requires the target learning material?*

**Evidence base.** Five internal reports and their underlying result files, cross-checked field-for-field against the source JSON prior to synthesis:

| Short name | File | Scope |
|---|---|---|
| **Failure Analysis** | `rq1_test_split_failure_analysis.md` (+ `rq1_flagged_questions.json`) | Manual failure taxonomy on 1,384 test items (SciQ 884, OBQA 500) |
| **Qwen3 Eval** | `kda_qwen3_4b_evaluation_report.md` (+ `kda_qwen3_4b_results.json`) | KDA scored with `Qwen3-4B-Instruct-2507` as the simulated student |
| **Counterfactual OBQA** | `counterfactual_obqa_analysis.md` (+ `results_counterfactual_sciq_test_full.json`, `results_counterfactual_obqa_test_full.json`) | Setting-C context perturbation, SciQ vs. OpenBookQA |
| **Persona Simulation** | `student_persona_simulation_report.md` (+ `results_persona_simulation_sciq.json`, `results_persona_simulation_obqa.json`) | Persona-conditioned zero-context roleplay as a saturation fix |
| **KDA Reproduction** | `kda_reproduction_summary.md` | Baseline `KDA_small` (355.1M-parameter, 4-model ensemble) reproduction |

All headline figures below were independently re-derived from the accompanying result JSONs (`n_flagged`, per-category counts, bucket percentages, denominators) and matched the prose reports exactly; no number in this synthesis is taken from prose alone.

---

## 1. Executive Verdict

**No. Modern LLM-based evaluators do not reliably measure whether a multiple-choice question requires its target learning material.** Across two datasets, two model scales (a 355.1M-parameter encoder ensemble and a 4B-parameter instruction-tuned LLM), two disentanglement methods (manual taxonomy and counterfactual context perturbation), and one direct mitigation attempt (persona-conditioned roleplay), the evidence converges on a single mechanism: **`KDA_cont`/`KDA_disc` conflate genuine contextual reliance with parametric priors, world common sense, and superficial option-construction heuristics.** The metric does not invert — it does not score context-independent questions systematically *lower* — it goes numerically *blind* to them, returning confident-looking scores computed from a shrinking and unrepresentative slice of the data as solver capability increases.

The failure is not a rare edge case. On the full SciQ test split (n = 884), the correlation between `KDA_cont` and zero-context solvability is **positive** (`r(KDA, P(R^q=1)) = +0.075`, Spearman ρ = +0.084) when a necessity metric must be strongly negative; on OpenBookQA (n = 500) it is **+0.301** (ρ = +0.370). At modern-LLM scale, the problem compounds: `Qwen3-4B-Instruct-2507` answers 95.36% of SciQ questions correctly with **no passage at all**, collapsing `KDA_disc`'s usable evidence base to **41 of 884 questions (4.6%)** while still reporting a superficially reassuring `KDA_disc = 0.9512`. Persona-conditioned prompting — the most direct attempt to manufacture a usable ability spread without external context — fails to recover a genuine necessity signal and instead degenerates into positional letter-fixation. The verdict holds independently on both datasets, through both scoring regimes, and it is corroborated, not merely asserted: 328/884 SciQ items (37.1%) and 80/500 OBQA items (16.0%) are independently confirmed answerable without the material by at least one of four disjoint detection criteria (§3).

---

## 2. Mathematical & Evaluator Mechanism Breakdown

### 2.1 Metric insensitivity, not inversion

The central diagnostic is what `KDA_cont` actually correlates with, computed on the **full, unannotated 884/500-item populations** (Failure Analysis, Table 7):

| | `r(KDA, P(R^q=1))` | `r(KDA, P(R^{q+f}=1))` | `r(KDA, gain)` |
|---|---:|---:|---:|
| SciQ | **+0.075** (ρ = +0.084) | +0.863 (ρ = +0.847) | +0.802 |
| OBQA | **+0.301** (ρ = +0.370) | +0.878 (ρ = +0.878) | +0.684 |

A metric measuring "does the material make this answerable" should correlate *strongly negatively* with zero-context success — the easier a question is without the material, the less the material should matter. Instead the correlation is weakly-to-moderately **positive**, and the metric tracks with-fact confidence (`r ≈ +0.86–0.88`) almost perfectly. This is confirmed non-parametrically at the decile level (Failure Analysis, Table 10): zero-context solve rate is **flat and non-monotone** across `KDA_cont` deciles, and the top decile — the questions the metric nominates as *most* knowledge-dependent — is still solved without any passage **41.6% of the time on SciQ and 42.0% on OBQA**. The flagged (context-independent) pool scores *higher*, not lower, than the rest of the split on both datasets: SciQ 0.491 vs. 0.460; OBQA 0.287 vs. 0.268. The `both_correct` bucket (questions solved with *and* without the fact) receives an above-average score on both datasets — SciQ 0.495 vs. a 0.471 split mean; OBQA 0.348 vs. 0.271, a +28% relative premium for questions demonstrably answerable without context.

### 2.2 Weight variance and denominator collapse

The mechanism is visible in the metric's own arithmetic. `KDA_cont(q) = Σₘ(1 − Pₘ(R^q=1))·Pₘ(R^{q+f}=1) / Σₘ(1 − Pₘ(R^q=1))` is a `(1 − P(R^q=1))`-weighted mean of with-fact confidence. When the weighting term is nearly constant across questions, the whole expression collapses onto an almost-unweighted mean of `P(R^{q+f}=1)` — a measure of *how easy the question is once the fact is supplied*, not of whether the fact was needed. On SciQ, two of the four `KDA_small` students are near-uniform on Setting A (Failure Analysis, Table 11):

| Model | weight mean | weight **sd** |
|---|---:|---:|
| `kda-albert-xlarge-v2-race` | 0.749 | **0.044** |
| `kda-scibert-uncased-race` | 0.718 | **0.091** |
| `kda-mpnet-base-race` | 0.591 | 0.256 |
| `t5-small-ssm-nq` | 0.616 | 0.397 |

Three of four students exhibit this near-uniform pattern on OBQA. With weight standard deviations in the **0.044–0.091** range against means near 0.72–0.75, the denominator barely moves from question to question, and the score is structurally pulled toward `P(R^{q+f}=1)` — which is precisely what is elevated for common-knowledge items, memorized terminology, stem-echo questions, and leaky option sets. This is corroborated directly: **29.8% (SciQ) and 30.8% (OBQA)** of the aggregate `KDA_cont` numerator mass is contributed by student–question pairs that were **already correct without the fact** (Failure Analysis, Table 8) — 256/884 SciQ questions (29.0%) and 133/500 OBQA questions (26.6%) draw more than half their numerator contribution from already-solved pairs.

### 2.3 Saturation at scale

The `KDA_small` ensemble (355.1M parameters total) is a weak-student baseline by design; the failure sharpens severely when the simulated student is replaced with a modern instruction-tuned model. `Qwen3-4B-Instruct-2507` (4-bit NF4) was scored on the same two full test splits (Qwen3 Eval):

| Dataset | $n$ | $Acc_{wof}$ (KDA_small, pooled vote) | $Acc_{wof}$ (Qwen3-4B) | $\Delta$ | $KDA_{disc}$ | Usable support |
|---|---:|---:|---:|---:|---:|---:|
| SciQ | 884 | 45.48% | **95.36%** | +49.88 pp | 0.9512 | **41/884 = 4.6%** |
| OBQA | 500 | 34.80% | **82.60%** | +47.80 pp | 0.5862 | **87/500 = 17.4%** |

$KDA_{disc}$'s denominator is $\sum_i(1-r_i^q)$ — the count of questions the solver got wrong without the fact. Every question the model already knows contributes **exactly zero** to numerator and denominator alike and is silently dropped. On SciQ, this leaves the reported $KDA_{disc}=0.9512$ resting on the statistical weight of **41 questions**, with **843 of 884 (95.4%)** landing in the invisible `both_correct` bucket. Nine out of ten (OBQA, `prior_share = 0.8891`) to nineteen out of twenty (SciQ, `prior_share = 0.9558`) of the model's with-fact-correct answers were **already correct before the fact was shown**. The continuous form offers no protection: $KDA_{cont}$'s denominator is 40.46/884 (4.58%) on SciQ and 86.22/500 (17.24%) on OBQA — a "quiet" version of the same collapse, since a denominator of 41 still divides cleanly and returns a number indistinguishable, on its face, from a well-supported score.

The probability distribution underlying this is close to bimodal: **777/884 SciQ samples (87.9%) and 257/500 OBQA samples (51.4%)** have $P(R^q{=}1)$ exactly `1.0` in float32; the middle probability band $[0.25,0.90)$ — where $KDA_{cont}$'s continuous relaxation could add information over the discrete form — holds only **0.23% of SciQ and 3.40% of OBQA**. Consequently $KDA_{cont} \approx KDA_{disc}$ (0.9407 vs. 0.9512 SciQ; 0.5956 vs. 0.5862 OBQA), and the continuous metric becomes an expensive re-derivation of the discrete one rather than a source of additional graded signal. The accuracy gain from the fact — the quantity KDA is meant to isolate — falls from **+43.10 pp** (KDA_small pooled vote, SciQ) to **+4.41 pp** (Qwen3-4B, SciQ) not because the material became less useful, but because there was almost nothing left for it to fix.

---

## 3. Empirical Failure Taxonomy (Test-Split Findings)

Every one of the 408 flagged items (union of four detection criteria — `KDA_cont ≥ 0.70`, discrete-KDA proxy `= 1.0`, `both_correct` at ≥0.70 confidence, and SciQ-only counterfactual `prior_dependent`) was manually read and assigned exactly one primary failure category.

| Failure category | SciQ *n* | SciQ % of split | 95% CI | OBQA *n* | OBQA % of split | 95% CI |
|---|---:|---:|---|---:|---:|---|
| `common_knowledge` | 182 | 20.6% | [18.1, 23.4] | 41 | 8.2% | [6.1, 10.9] |
| `parametric_knowledge` | 48 | 5.4% | [4.1, 7.1] | 0 | 0.0% | [0.0, 0.8] |
| `option_leakage` | 45 | 5.1% | [3.8, 6.7] | 13 | 2.6% | [1.5, 4.4] |
| `reasoning_shortcut` | 44 | 5.0% | [3.7, 6.6] | 16 | 3.2% | [2.0, 5.1] |
| `material_not_necessary` | 9 | 1.0% | [0.5, 1.9] | 10 | 2.0% | [1.1, 3.6] |
| **Total flagged** | **328** | **37.1%** | [34.0, 40.3] | **80** | **16.0%** | [13.0, 19.5] |

*(Verified against `rq1_flagged_questions.json`: `n_flagged`/`split_size` and per-category counts reproduce these figures exactly.)*

**The datasets fail for structurally different reasons.** SciQ's gold answer string appears verbatim in its target passage for 838/884 items (94.8%), and 494/884 (55.9%) are near-verbatim cloze deletions of a single passage sentence — SciQ leaks *specialized terminology and distractor construction*. OpenBookQA's gold answer appears in its one-clause `fact1` for only 57/500 items (11.4%) — it has **zero** `parametric_knowledge` failures in the flagged pool, because there is no specialized vocabulary to memorize; it instead leaks *common sense and answer shape*, including a statistically significant longest-option bias (160/427 = 37.5% vs. 25% chance, z = 5.95) that is **absent on SciQ** (184/695 = 26.5%, z = 0.90, n.s.).

**Representative cases** (all verified against `rq1_flagged_questions.json`):

- **SciQ #587** — *"What are catalysts in living things called?"* Options include **"carbohydrates" twice**, making the item 1-of-3 by construction. `KDA_cont = 0.8419` is the **single highest score in the entire SciQ test split**, and the counterfactual run independently confirms the diagnosis: when the passage is rewritten to assert a different answer, the ensemble still answers "enzymes" (`prior_dependent`, `kda_adjusted_hard = 0.208`). The metric's own top-ranked "most knowledge-dependent" item is a broken item solved from priors.
- **SciQ #650** — *"What disease is the result of unchecked cell division…?"* (gold: `cancer`, rank 12/884 by KDA, `KDA_cont = 0.7420`). The primary model scores 0.792 with no passage at all; the metric reads the residual jump to 0.999 as strong dependency when it is headroom, not dependency (`common_knowledge`).
- **OBQA #375** — *"If you find something smooth and hard on the ground, it is probably made of what?"* (gold: `minerals`, rank 3/500, `KDA_cont = 0.6203`). Physical-state elimination ("smooth and hard") removes three of four options before any mineralogy is consulted (`common_knowledge`).
- **SciQ #121** — *"gibberellins"* as the plant growth hormone: a 110M-parameter encoder that has never seen the passage assigns it 0.649 from pretraining alone (`parametric_knowledge`, rank 27/884).
- **OBQA #339** — *"How do polar bears survive the cold?"* Gold option is the literal string **"B and D"**, structurally unlike every distractor — the highest zero-context ensemble confidence in the whole OBQA split (0.765); adding the fact made the ensemble *worse* (0.538) (`option_leakage`).
- **SciQ #162** — *"Digestive enzymes are… secreted by… which body system?"* The stem contains the literal word "Digestive"; the gold option is the only one that repeats it — pure lexical matching, no physiology engaged (`reasoning_shortcut`).
- **SciQ #291** and **OBQA #15** — arithmetic-progression and tautological-fact items respectively, where the stem alone forces the answer and the "fact" supplies nothing the question did not already determine (`material_not_necessary`).

Structural-cue enrichment inside the flagged pool is honestly reported as **weak** on SciQ (risk ratios 1.13–1.21 for templated option families and longest-option bias) — these cues are properties of the datasets at large, not concentrated in the specific items the models exploit. What is robust, independent of any manual label, is the population-level insensitivity finding in §2.1–2.2, which requires no annotation at all.

---

## 4. Counterfactual & Simulation Corroboration

### 4.1 Setting-C disentanglement

Counterfactual context perturbation (rewriting the target fact so it asserts a *different*, incorrect answer) provides an annotation-free check on whether a `both_correct` item's success is genuinely tied to the material. On SciQ, where the extractive `support` passage makes answer-span substitution a clean one-word swap in 813/884 items (92.0% `exact` tier, 860/884 = 97.3% eligible overall), the manipulation does real work: the primary model flips from 0.083 accuracy on the true gold answer to 0.853 on the counterfactual target. **13.4% of counterfactual-eligible SciQ items (115/860) are `prior_dependent`** — the ensemble keeps the gold answer even when the passage is rewritten — and their mean *unadjusted* `KDA_cont` (0.474) is statistically indistinguishable from the split mean (0.477), while the *adjusted* score collapses to 0.208.

Within the `both_correct` pool specifically — the cell where Setting C is the only way to tell whether the fact mattered — prior-dependence ranges widely by dataset, model, and match tier:

| | SciQ | OBQA (`partial`, n=145 eligible) | OBQA (`exact`, n=42 eligible) |
|---|---:|---:|---:|
| Primary model, `both_correct` `prior_dependent` | **8.2%** (n=415) | 46.2% (n=52) | 16.7% (n=12) |
| Ensemble, `both_correct` `prior_dependent` | 22.3% (n=381) | 41.5% (n=41) | **50.0%** (n=16) |

Across these conditions, the fraction of with-fact-correct answers that are demonstrably **prior-driven rather than context-driven** spans roughly **8% to 50%**, and OpenBookQA is consistently 2–3× worse than SciQ on this axis. This asymmetry is corroborated at the accuracy level: raw `Acc_wf` on OBQA-eligible items (0.5793) overstates genuine context use by **16.6 points** once prior-driven correct answers are stripped (`Acc_wf^verified = 0.3517`) — **4.2× the 4.0-point inflation measured on SciQ** (0.9058 → 0.8581).

OpenBookQA's counterfactual mechanism itself is degraded relative to SciQ's, and this is reported transparently: only 145/500 OBQA items (29.0%) are eligible for span substitution at all (vs. 860/884 = 97.3% on SciQ), because `fact1` is a *deductive* one-clause rule ("predators eat prey") related to its answer by inference rather than by string identity, unlike SciQ's *extractive* passage. 67% of the eligible OBQA pool falls in the low-fidelity `partial` tier (81% producing ungrammatical splices). Despite this caveat, the **direction of the finding is stable across both the degraded (`partial`) and clean (`exact`) subsets** — OBQA's prior-dependence rate is elevated under both — which is why it is treated as the more trustworthy of the counterfactual conclusions. A secondary diagnostic reinforces the extractive/deductive split: on SciQ, the model's Setting-C prediction matches the highest-lexical-overlap option 86.5% of the time (largely string-copying, not comprehension); on OBQA this drops to 31.0% (47.6% on the clean `exact` tier) because the rule-to-instance inference stands between the fact and the answer, and the models mostly cannot make that inferential jump — which is precisely why Setting C loses diagnostic power on OBQA and inflates the residual `unstable_other` class (37.2% on OBQA vs. 6.4% on SciQ, primary model).

### 4.2 Why prompting fails as a fix

Persona Simulation tests whether asking `Qwen3-4B-Instruct-2507` to roleplay Beginner / Intermediate / Advanced students can manufacture the ability spread that saturation has erased, without any external context. It does not.

**Isolated (independent) roleplay is a no-op with respect to saturation.** The Advanced persona reproduces the persona-free baseline almost exactly (SciQ 0.956 vs. 0.954; OBQA 0.824 vs. 0.826), and even the Beginner persona barely moves accuracy (SciQ 0.920, a 3.4-point drop). Total spread across all three tiers is **0.036 on SciQ — narrower than the width of its own 95% confidence intervals** — and 0.108 on OBQA. The persona is monotone (Beginner < Intermediate < Advanced) and, uniquely among the two paradigms tested, produces a properly *nested* error structure (76.9% of Advanced errors on SciQ are also Beginner errors; 72.7% on OBQA), but it simply leaves the parametric prior intact.

**Joint (contrastive) prompting produces large gaps that are not ability — they are an emission-order artifact.** With tiers scored in a shared context (Beginner → Intermediate → Advanced), SciQ shows a striking +0.342 gap and OBQA shows a **non-monotone, inverted** −0.228 gap (Advanced *below* Beginner). Reversing the emission order (Advanced → Intermediate → Beginner) **flips the direction of the effect on both datasets**: SciQ's gap widens to +0.639 and OBQA's inverts to +0.408, both becoming monotone. Because all three tiers give distinct answers only 5–15% of the time (chance for independent uniform draws would be 37.5%), the tiers are not independent judgments — they are one shared answer plus a positional perturbation. Whichever tier is written **last** is pressured to disagree with the tiers already committed to context: 71.4% of the time on SciQ (descending order) and up to 92.1% of those breaks are *away from a correct consensus*. **The joint paradigm does not simulate ability; it simulates disagreement, and the ability label only determines who is unlucky enough to be scored last.**

**Neither paradigm reproduces the designed failure modes.** No beginner trap-rate (longest-option preference, lookalike-to-gold confusion) significantly exceeds the 1/3 chance baseline for a uniform-over-distractors error pattern (all *p* ≥ 0.10, one case significantly *below* chance at *p* = 0.02). What actually happens under degradation pressure is **positional collapse**: in the SciQ ascending-order joint run, 63% of the suppressed Beginner tier's errors land on a single letter ("A", χ² = 346.1, *p* < 0.001 against the dataset's own gold-letter distribution); in the OBQA descending-order run, 61% land on "D" (χ² = 339.6, *p* < 0.001). The one condition that avoids letter fixation collapses instead to chance accuracy (0.291, chance = 0.25). Persona prompting degrades a saturated model into either a fixed-letter guesser or a coin-flipper — never into a plausible weaker reader.

**Implication.** Persona-conditioned zero-context simulation is not a usable substitute for the ability spread that saturation destroys. A difficulty ranking built on the joint tiers would primarily rank *option position*; one built on the isolated tiers would carry almost no dynamic range. Restoring a usable necessity signal on a saturated model requires a mechanism that genuinely removes knowledge — context ablation, counterfactual passages, or an actually weaker student model — not one that asks a capable model to pretend.

---

## 5. Implications for AQG Evaluation & Next Steps

**The practical consequence for automatic question generation (AQG) is a reward-shaping problem, not just a measurement nuance.** Because `KDA_cont`/`KDA_disc` are positively (not negatively) associated with context-independence, and because the high-scoring tail of both test splits is *enriched* for flagged items rather than depleted of them (top-decile SciQ: 45.5% flagged vs. a 37.1% base rate; `KDA_cont ≥ 0.70`: 51.4% flagged), an AQG system evaluated or optimized against this metric is systematically rewarded for generating **trivia-adjacent, memorizable, or structurally leaky questions** — common-knowledge stems, terminology recoverable from pretraining, duplicated or shape-distinct distractors, cloze deletions that echo the stem — while a genuinely novel, context-dependent, pedagogically well-designed question is scored no higher, and sometimes lower, than these degenerate cases. The metric's one reliable positive signal — the `wrong_to_correct` bucket, where the fact produces a decisive flip from wrong to right (39/884 SciQ, 51/500 OBQA at Qwen3-4B scale; 4.6%/17.4% of each split) — is exactly the population an AQG evaluator should be selecting *for*, and it is the population current scoring makes hardest to isolate as solver capability rises, since it is diluted by an ever-growing `both_correct` bucket that the discrete metric discards and the continuous metric barely down-weights.

This risk compounds with model scale rather than attenuating: as the simulated student or evaluator model becomes more capable (355M-parameter encoder → 4B-parameter instruction-tuned LLM), the usable evidentiary base for a necessity judgment shrinks from a workable-but-imperfect ensemble reading to **4.6% of SciQ and 17.4% of OpenBookQA**, precisely because a stronger evaluator is more likely to already know the answer — the same dynamic the field will face using ever-more-capable LLMs as automatic judges. A benchmark or leaderboard built on this metric would silently favor AQG systems that produce content the *evaluator* already knows, which is the opposite of the intended incentive for a pedagogically useful question generator.

**This motivates a shift from correlational KDA scoring toward counterfactual-adjusted and causal evaluation paradigms.** The counterfactual-adjustment machinery already prototyped in the SciQ/OBQA Setting-C work — `KDA_adj^hard`, `KDA_adj^soft`, `KDA_adj^excl` — demonstrates the core idea (discount or exclude `both_correct` items whose fact-dependence cannot survive a perturbation test) but also demonstrates its current limits: the gate presupposes a well-formed counterfactual, which holds for 97.3% of SciQ but only 29.0% of OpenBookQA, because answer-span substitution is the wrong primitive for a rule-to-instance (deductive) benchmark. The clear next step, identified directly in the evidence base, is **rule-level counterfactual perturbation** rather than answer-string substitution — negating or antonym-substituting the predicate of a deductive fact (e.g., "a plant requires sunlight to grow" → "…requires darkness to grow") so that grammaticality and coverage no longer depend on the answer appearing verbatim in the source text — validated by an explicit, reported "is this counterfactual well-formed and does some option remain correct under it?" check before any downstream metric is computed on top of it.

Two further directions follow directly from what has already failed here and should be treated as out of scope for further "fix the prompt" attempts: (1) **persona-based or roleplay-based mitigation of evaluator saturation is not viable** — it produces either a no-op (isolated) or a positional artifact masquerading as ability (joint) — so any RQ2/RQ3 metric redesign should assume the evaluator's raw zero-context capability is a fixed, unavoidable confound to be adjusted for statistically (as in §4.1), not prompted away; and (2) **cross-dataset KDA comparisons require controlling for target-fact style** (extractive vs. deductive), since SciQ and OBQA are not on a comparable scale even before any of the failure modes above are considered (KDA Reproduction: mean KDA 0.4715 vs. 0.2712, a gap shown to be substantially a mixture effect between 92.8%-verbatim extraction and 90.6%-deductive items). A causal evaluation framework for RQ2/RQ3 should therefore report evaluator zero-context accuracy alongside any dependency score as a mandatory validity check, apply counterfactual (or, for deductive material, rule-level) adjustment before treating a `both_correct` item as evidence of necessity, and stratify all cross-dataset claims by fact-construction type rather than comparing raw KDA values directly.

---

## Appendix — Study-Level Limitations Carried Forward

These caveats, drawn directly from the source reports, bound the strength of the claims above and should travel with any downstream use of this synthesis:

- **Single annotator, no inter-rater agreement** on the 408-item failure taxonomy (§3); the four-way split among `common_knowledge` / `parametric_knowledge` / `reasoning_shortcut` / `option_leakage` is the most annotator-sensitive part of the study. The population-level insensitivity findings (§2.1–2.2) do **not** depend on these labels and rest on the full, unannotated 884/500-item populations.
- **`KDA_disc` throughout is a model-ensemble proxy**, not a metric computed over real human-student responses; no human-subject data exists in this evidence base.
- **The OBQA counterfactual pool (§4.1) is a 29.0% non-random, extraction-biased subset** of the split, and 67% of it is low-fidelity (`partial`-tier) substitution; the SciQ-vs-OBQA prior-dependence comparison is directionally robust across degraded and clean subsets but should not be read as precisely calibrated.
- **Criterion C3 (counterfactual `prior_dependent`) could not be applied to OpenBookQA in the main taxonomy (§3)**, making the 37.1% (SciQ) vs. 16.0% (OBQA) flagged rates not directly comparable — SciQ had three independent detection criteria available, OBQA only two.
- **The Qwen3-4B evaluation (§2.3) is a single solver, single seed, single quantization**, with no option-order debiasing and no independent calibration check on the 4-bit NF4 logits; the support-collapse finding itself follows from `Acc_wof` alone and does not depend on these choices, but the exact KDA values would shift under a different configuration.
- **Persona Simulation (§4.2) covers one model, one quantization**, and only two of six possible tier emission orders; the positional-collapse and chance-accuracy findings are the load-bearing results and do not depend on the specific trap-detection heuristics used.
- All relationships reported here are **correlational**; no attention- or attribution-level analysis isolates *why* any individual model answered as it did.
