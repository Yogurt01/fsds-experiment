# Counterfactual Context Perturbation (Setting C) on OpenBookQA

Companion to the SciQ counterfactual run. This report covers the Setting-C experiment on the
**OpenBookQA test split** and the head-to-head comparison with SciQ, whose numbers are quoted
throughout from the pre-existing, unmodified run.

| | |
|---|---|
| Datasets | `datasets/openbookqa/obqa_test_full.json` (500 q) · `datasets/sciq/sciq_test_full.json` (884 q) |
| Simulated students | `KDA_small` (\|M\| = 4): `t5-small-ssm-nq`, `kda-albert-xlarge-v2-race`, `kda-mpnet-base-race`, `kda-scibert-uncased-race` |
| Primary model | `Riiid/kda-mpnet-base-race` |
| Settings | A = question + options · B = + gold fact · C = + counterfactual fact |
| Script | `code/ex2_counterfactual/run_counterfactual_experiment.py` |
| Device | CUDA (RTX 3050 Laptop, 4 GB), 4 models loaded sequentially |
| Runtime | 243.2 s (0.478 s/question) for the primary run |

**Outputs**

```
results/ex2_counterfactual/results_counterfactual_obqa_test_full.json        primary, --min-substitution-tier partial
results/ex2_counterfactual/counterfactual_obqa_test_full.log
results/ex2_counterfactual/results_counterfactual_obqa_test_exact_tier.json  sensitivity, --min-substitution-tier exact
results/ex2_counterfactual/counterfactual_obqa_test_exact_tier.log
results/ex2_counterfactual/counterfactual_passages_preview_obqa.json         generation preview (dry run)
results/ex2_counterfactual/counterfactual_obqa_dryrun.log
```

---

## 0. Headline

The experiment ran cleanly and produced every requested metric, but **the counterfactual
mechanism itself does not transfer from SciQ to OpenBookQA**. Answer-span substitution assumes
the reference material *states* the answer; OpenBookQA's `fact1` is a general rule of which the
gold option is an instance, so 71% of the split has no span to substitute and two thirds of what
survives is substituted badly. The resulting Setting C moves the models barely at all
(`counterfactual_passage_vs_gold` 0.310 vs `counterfactual_passage_vs_cf_target` 0.317 on the
primary model), and the adjusted metrics inherit that noise.

The one comparison that survives the caveat, because it points the same way on both the degraded
and the clean subset, is the `both_correct` pool: **prior-driven correct answers roughly double
on OpenBookQA relative to SciQ** (41.5% vs 22.3% at ensemble level; 50.0% on the clean `exact`
subset).

---

## 1. Eligibility and lexical failure

### 1.1 Tier distribution

A sample is counterfactual-eligible when the gold answer can be located in the passage and
replaced by a distractor. Four matching tiers are tried in order (`exact`, `glued`,
`morphological`, `partial`); `none` means no tier matched.

| Tier | SciQ (884) | OpenBookQA (500) |
|---|---:|---:|
| `exact` | 813 · **92.0%** | 42 · **8.4%** |
| `glued` | 0 · 0.0% | 1 · 0.2% |
| `morphological` | 11 · 1.2% | 5 · 1.0% |
| `partial` | 36 · 4.1% | 97 · **19.4%** |
| `none` (ineligible) | 24 · 2.7% | 355 · **71.0%** |
| **Eligible** | **860 · 97.3%** | **145 · 29.0%** |

### 1.2 Why the mechanism fails on rule-to-instance facts

SciQ's `support` is an **extractive** paragraph lifted from a textbook: the sentence that answers
the question contains the answer string verbatim, so deleting it and inserting a distractor is a
faithful, well-formed perturbation.

OpenBookQA's `fact1` is a **deductive** one-clause rule. The question asks for an *instance*; the
fact supplies the *general principle*. The two are related by inference, not by string identity:

| Fact (`fact1`) | Question asks | Gold option | Overlap |
|---|---|---|---|
| `predators eat prey` | what a fox might eat | `bunnies` | none |
| `using less resources usually causes money to be saved` | best way to save money | `quit eating lunch out` | none |
| `fog is formed by water vapor condensing in the air` | where fog forms | `a marsh` | none |

There is nothing to substitute. This accounts for the 355 `none` cases (71.0%) directly, and it
is a property of the benchmark's design, not a defect in the generator.

### 1.3 The surviving 145 are mostly low-fidelity

Fidelity of the substitution, measured on the eligible subset. *Coverage* is the fraction of the
gold answer's content words that the matched span actually covered.

| Tier | n | coverage = 1.0 | coverage < 0.5 | inserted target differs from matched span by > 1 token |
|---|---:|---:|---:|---:|
| `exact` | 42 | 100.0% | 0.0% | 0.0% |
| `glued` | 1 | 100.0% | 0.0% | 0.0% |
| `morphological` | 5 | 100.0% | 0.0% | 0.0% |
| `partial` | 97 | **1.0%** | **79.4%** | **81.4%** |
| **All eligible** | **145** | 33.8% | 53.1% | 54.5% |

The `partial` tier is **67% of the eligible pool** and matches only a fragment, then splices an
entire distractor option into the middle of the clause. The result is usually not a sentence:

| id | `fact1` → counterfactual |
|---|---|
| 123 | `An example of playing a musical instrument is strumming a guitar string` → `… is strumming a guitar hit a toy baseball with a bat` |
| 15 | `as the use of alternative fuels increases , the use of gasoline will decrease` → `… the use of wind power will be expensive decrease` |
| 74 | `if a leaf falls off of a tree then that leaf is dead` → `if a leaf falls off of a tree then is likely to continue to grow leaf is dead` |
| 460 | `moving an object from a cool place to a warm place causes …` → `moving an object an ice tray is placed in a freezer a cool place to a warm place causes …` |
| 185 | `an landfill is a source of pollution` → `an a man who lives in a great suburb is a source of pollution` |

The single `glued` match is a false positive: the gold answer `"O"` (chemical symbol) matched the
letter *o* inside "to", producing `the respiratory system transfers oxygen twater the circulatory
system`.

**Net:** on SciQ the counterfactual is a clean one-word swap in 92.0% of the split. On OpenBookQA
it is clean in 8.4%, degraded in 19.4%, and impossible in 71.0%.

---

## 2. Accuracy across Settings A / B / C, and metric degeneration

### 2.1 Per-model accuracy

`C vs gold` = still answers the true gold with the perturbed fact. `C vs cf-target` = follows the
perturbed fact. `Acc_wf^verified` = Setting-B accuracy after stripping correct answers that
Setting C shows to be prior-driven.

| Dataset | Model | A | B | C vs gold | C vs cf-target | Acc_wf^verified |
|---|---|---:|---:|---:|---:|---:|
| SciQ | `t5-small-ssm-nq` | 0.398 | 0.630 | 0.328 | 0.422 | 0.394 |
| SciQ | `kda-albert-xlarge-v2-race` | 0.265 | 0.624 | 0.193 | 0.547 | 0.566 |
| SciQ | **`kda-mpnet-base-race`** | 0.514 | **0.897** | **0.083** | **0.853** | **0.858** |
| SciQ | `kda-scibert-uncased-race` | 0.361 | 0.678 | 0.166 | 0.621 | 0.587 |
| SciQ | ensemble (pooled vote) | 0.455 | 0.886 | 0.134 | 0.778 | 0.785 |
| OBQA | `t5-small-ssm-nq` | 0.238 | 0.294 | 0.117 | 0.676 | 0.317 |
| OBQA | `kda-albert-xlarge-v2-race` | 0.300 | 0.406 | 0.331 | 0.276 | 0.276 |
| OBQA | **`kda-mpnet-base-race`** | 0.434 | **0.528** | **0.310** | **0.317** | **0.352** |
| OBQA | `kda-scibert-uncased-race` | 0.322 | 0.354 | 0.366 | 0.255 | 0.234 |
| OBQA | ensemble (pooled vote) | 0.348 | 0.406 | 0.179 | 0.662 | 0.379 |

### 2.2 Why Setting C degenerates into noise

On SciQ the primary model flips decisively when the passage is perturbed: 0.083 on the gold
answer, 0.853 on the counterfactual target. The manipulation is doing real work, and the
`context_dependent` / `prior_dependent` split is meaningful.

On OpenBookQA the same model lands at 0.310 / 0.317 — an almost even split. It neither holds the
gold answer nor follows the perturbed fact. Three causes compound:

1. **Setting B was already weak.** The gold fact only buys +0.094 accuracy on OBQA (0.434 → 0.528)
   against +0.383 on SciQ (0.514 → 0.897). A perturbation cannot remove a dependency that was
   never established.
2. **The perturbed fact is often ungrammatical**, so there is no coherent proposition to follow.
3. **The rule-to-instance step is beyond these students.** Even the *unperturbed* fact requires an
   inference the 110M-parameter encoders largely cannot make.

The `unstable_other` rate quantifies the degeneration directly: on the primary model it is **37.2%**
of eligible OBQA items — the Setting-C prediction is neither gold nor counterfactual target, it
simply scatters — against **6.4%** on SciQ.

`t5-small-ssm-nq` is the informative outlier (`C vs cf-target` = 0.676, far above the other three).
It is a generative QA model that copies surface spans from its prompt, so it follows the
spliced-in distractor text even when the host sentence is word salad. That is span-copying, not
context sensitivity.

### 2.3 Adjusted KDA

| Metric | SciQ (n=860) | **OBQA `partial` (n=145)** | OBQA `exact` (n=42) |
|---|---:|---:|---:|
| `KDA_cont`, all samples | 0.4715 | 0.2712 | 0.2712 |
| `KDA_cont`, CF-eligible | 0.4767 | 0.3047 | 0.3812 |
| **`KDA_adj^hard`** | 0.3357 | **0.1275** | 0.1902 |
| **`KDA_adj^soft`** | 0.2641 | **0.1145** | 0.1734 |
| **`KDA_adj^excl`**, zero-filled | 0.4269 | **0.2580** | 0.3057 |
| **`KDA_adj^excl`**, retained mean (README §2.4 form) | **0.4793** | **0.2969** | **0.3891** |
| retention, hard | 70.4% | **41.8%** | 49.9% |
| retention, soft | 55.4% | **37.6%** | 45.5% |
| retention, sample-exclusion (zero-filled) | 89.6% | 84.7% | 80.2% |
| retention, sample-exclusion (retained mean) | **100.5%** | **97.4%** | **102.1%** |
| dropped (prior-dependent + unstable) | 94 / 860 | 19 / 145 | 9 / 42 |
| mean `KDA_cont` of the dropped items | 0.4554 | 0.3568 | 0.3522 |
| AUC, `KDA_cont` separating kept from dropped | 0.553 | **0.346** | 0.566 |

Gate-based adjustment removes **58% of OpenBookQA's KDA mass** against 30% of SciQ's. That gap is
mostly an artifact rather than a finding: `KDA_adj^hard` requires the model to *follow* the
counterfactual, and on OBQA the counterfactual is frequently something no sensible reader would
follow. The gate cannot distinguish "ignored the context" from "declined to follow word salad".

`KDA_adj^excl` is the more trustworthy of the three on this dataset, because it only drops items
where the model demonstrably *kept* the gold answer under perturbation — a condition that does not
depend on the counterfactual being well-formed. But read as README §2.4 defines it, a mean over the
*retained* set, it barely moves the score on any of the three runs (97.4%–102.1% retention): the
dropped items are not low-`KDA_cont` items. On OBQA `partial` they score *higher* than the items
kept (0.3568 vs 0.2969), so the AUC is inverted at 0.346 — a higher `KDA_cont` there mildly predicts
a failed context check. The 84.7% figure is the zero-filled form and is dominated by the keep rate
(126/145 = 86.9%). See [`adjusted_kda_corrected.json`](../../results/ex2_counterfactual/adjusted_kda_corrected.json)
for both readings and the $\mathcal{Q}_{ctx}$ sensitivity sweep.

---

## 3. Prior vs. context dependency

### 3.1 All eligible items

Ensemble classification:

| Class | SciQ (860) | OBQA `partial` (145) | OBQA `exact` (42) |
|---|---:|---:|---:|
| `context_dependent` | 669 · 77.8% | 96 · 66.2% | 24 · 57.1% |
| `prior_dependent` | 115 · 13.4% | 26 · **17.9%** | 10 · **23.8%** |
| `unstable_other` | 76 · 8.8% | 23 · **15.9%** | 8 · **19.0%** |

Primary model (`kda-mpnet-base-race`):

| Class | SciQ | OBQA `partial` |
|---|---:|---:|
| `context_dependent` | 734 · 85.3% | 46 · **31.7%** |
| `prior_dependent` | 71 · 8.3% | 45 · **31.0%** |
| `unstable_other` | 55 · 6.4% | 54 · **37.2%** |

### 3.2 The `both_correct` pool

This is the RQ1-relevant cell: questions answered correctly *both* with and without the fact,
where Setting C is the only way to tell whether the fact mattered.

| | SciQ | OBQA `partial` | OBQA `exact` |
|---|---:|---:|---:|
| **Primary model** — n | 415 | 52 | 12 |
| `context_dependent` | 90.1% | **36.5%** | 58.3% |
| `prior_dependent` | 8.2% | **46.2%** | 16.7% |
| `unstable_other` | 1.7% | 17.3% | 25.0% |
| **Ensemble** — n | 381 | 41 | 16 |
| `context_dependent` | 75.3% | 53.7% | 43.8% |
| `prior_dependent` | 22.3% | **41.5%** | **50.0%** |
| `unstable_other` | 2.4% | 4.9% | 6.3% |

Among questions the models get right in both settings, OpenBookQA items are roughly **twice as
likely to be prior-driven** as SciQ items (ensemble 41.5% vs 22.3%), and the clean `exact` subset
is worse rather than better (50.0%). Because the degraded and clean subsets agree in direction,
this is the finding least contaminated by the substitution-quality problem.

### 3.3 Context-verified accuracy

`Acc_wf^verified = (wrong_to_correct + context-verified both_correct) / n_eligible`;
prior inflation `= n_prior_dependent_in_both_correct / n_eligible`.

| | SciQ | OBQA `partial` | OBQA `exact` |
|---|---:|---:|---:|
| raw `Acc_wf` on eligible | 0.9058 | 0.5793 | 0.5476 |
| **`Acc_wf^verified`** | **0.8581** | **0.3517** | 0.4286 |
| prior inflation | 0.0395 | **0.1655** | 0.0476 |

Setting-B accuracy on OpenBookQA overstates genuine context use by **16.6 points** once
prior-driven correct answers are stripped — **4.2× the 4.0-point inflation on SciQ**. The `exact`
subset gives 4.8 points, but on n = 42 with only 12 `both_correct` items that figure carries
almost no weight on its own.

---

## 4. Extractive vs. deductive: a string-overlap diagnostic

To test whether "following the counterfactual" reflects comprehension or string matching, each
eligible item was checked for whether the primary model's Setting-C prediction is simply the
option with the highest content-word overlap with the perturbed fact.

| Dataset | Tier | n | pred_C = max-overlap option | cf-target = max-overlap option |
|---|---|---:|---:|---:|
| SciQ | all | 860 | **86.5%** | 98.7% |
| SciQ | `exact` | 813 | 86.5% | 98.8% |
| OBQA | all | 145 | **31.0%** | 98.6% |
| OBQA | `exact` | 42 | 47.6% | 100.0% |
| OBQA | `partial` | 97 | 23.7% | 99.0% |

In ~99% of items on **both** datasets the counterfactual target is the string most present in the
perturbed fact, so the lexical cue is equally available everywhere. SciQ models take that cue
86.5% of the time; OpenBookQA models only 31.0%.

Two conclusions follow:

- On SciQ, "context-dependent" substantially means **copies the answer out of the passage**. The
  high `context_dependent` rate (77.8%) is partly a measure of extractive string matching, not of
  comprehension. This qualifies the SciQ result as much as it explains the OBQA one.
- On OpenBookQA, the models cannot exploit the cue because the rule-to-instance inference stands
  between the fact and the option. Setting C therefore loses its diagnostic power, and the
  elevated `prior_dependent` and `unstable_other` rates are what a failed manipulation looks like
  from the inside.

A tier-stratified check adds a caution that runs against intuition: `prior_dependent` is **higher**
on the clean `exact` tier (23.8%) than on the degraded `partial` tier (14.4%). The likely reason
is that the spliced-in full distractor phrase in `partial` items acts as its own lexical lure —
inflating apparent context-dependence rather than prior-dependence. This is reported as observed;
the current design cannot separate the two mechanisms.

---

## 5. Limitations

1. **The OBQA counterfactual pool is a 29% non-random subset.** Eligibility requires the fact to
   lexically contain the answer, which selects for the most SciQ-like (extractive) OBQA items.
   Every OBQA figure here generalises to that subset, not to the split.
2. **67% of the eligible pool has degraded substitutions** (`partial` tier, ~81% ungrammatical).
   The `exact`-tier run is included as a sensitivity check; it agrees on direction for the
   `both_correct` finding and disagrees on prior inflation.
3. **The `exact` subset is n = 42**, with 12 `both_correct` items for the primary model. Direction
   only; no interval on it would be meaningful.
4. **`prior_dependent` conflates two behaviours on OBQA** — genuinely ignoring context, and
   sensibly declining to follow an incoherent string. Not separable under the current design.
   The same objection applies with more force to `unstable_other`, which is **33.8% of OBQA
   `partial` (model, question) pairs and 34.5% of `exact` pairs** — far above SciQ's 19.7%.
   Since 2026-09-09 all four estimators score these as failures (the strict convention,
   [`unstable_other_convention.md`](unstable_other_convention.md)), so every OBQA figure in
   this document is depressed by perturbation quality as well as by solver behaviour. The
   alternative — crediting them — would have raised OBQA's hard retention from 41.8% to
   71.0% and erased most of the SciQ-vs-OBQA gap this report documents, which is why it was
   rejected.
5. **`KDA_adj^hard` / `KDA_adj^soft` are not comparable across the two datasets**, because the gate
   presupposes a well-formed counterfactual and OBQA frequently lacks one. `KDA_adj^excl` is the
   only variant that supports a cautious cross-dataset reading — but note §2.3: in its retained-mean
   form it is close to inert on both datasets, so what it supports is a comparison of *how many*
   items each dataset sheds, not of how much their scores change.
6. **The SciQ side of every comparison is the pre-existing run, unchanged.**

### Script changes required to run at all

`code/ex2_counterfactual/run_counterfactual_experiment.py` was SciQ-specific and could not
execute on OpenBookQA. Five minimal, backward-compatible edits were applied; none touch scoring
logic:

| Location | Change | Reason |
|---|---|---|
| `score_sample` return | `sample["sciq_index"]` → additive `sciq_index` / `obqa_index` / `source_index` keys | hard `KeyError` on OBQA samples; SciQ schema preserved |
| examples table | log `source #%s` instead of `SciQ #%s` | same `KeyError` |
| summary block | `"dataset": "allenai/sciq"` → derived from the data filename | output would have been mislabelled |
| `--dry-run` | preview path derived from `--out` instead of a fixed filename | the fixed path overwrote the SciQ preview on any non-SciQ dry run |

---

## 6. Future direction: semantic perturbation for deductive benchmarks

Answer-span substitution is the wrong primitive for a rule-to-instance benchmark. It is bounded
above by 29% coverage on OpenBookQA and yields a well-formed perturbation for only 8.4% of the
split. Raising coverage requires perturbing the **rule**, not the answer string.

Three candidate strategies, in increasing order of cost:

1. **Rule-level negation / polarity inversion.** `predators eat prey` → `predators avoid prey`;
   `a plant requires sunlight to grow` → `a plant requires darkness to grow`. Applies to nearly
   every `fact1` because it targets the predicate rather than the answer, and preserves
   grammaticality by construction. The counterfactual target then has to be re-derived — the
   correct answer *under the inverted rule* is not simply a distractor, so this needs an
   answer-remapping step, and for some items no option is consistent with the inverted rule.
2. **Antonym / scalar substitution on the rule's head term**, using a controlled lexicon
   (`increase` ↔ `decrease`, `positive` ↔ `negative`, `hot` ↔ `cold`). Narrower coverage than
   negation but preserves sentence shape exactly and keeps the mapping to a distractor tractable.
3. **Generated counterfactual rules from an instruction-tuned model**, constrained to a single
   clause and validated by round-trip entailment. Highest coverage and highest risk; would need
   its own validation pass before any metric is computed on top of it.

Whichever is chosen, the validity check should be explicit and reported alongside the metric:
*a counterfactual is usable only if some option is correct under the perturbed rule and the
perturbed rule is a well-formed clause.* The current `substitution_tier` field is the right place
to record it. Neither this report nor the RQ1 diagnostic proposes a replacement for `KDA_cont`
itself — that remains out of scope.

---

## Related

- `docs/ex5_failure_audit/rq1_test_split_failure_analysis.md` — RQ1 diagnostic on both test splits, including the
  SciQ `prior_dependent` cohort used above.
- `docs/ex1_reproduce_KDA/kda_reproduction_summary.md` — the Setting A / B baselines quoted in §2.1.
