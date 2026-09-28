# Next-Phase Handoff

**Written:** 2026-09-08, at the end of the archival/reorganization session.
**Prerequisite reading:** [`PROJECT_READING_GUIDE.md`](../PROJECT_READING_GUIDE.md) — this file
assumes it and does not repeat it.
**Scope:** the three items requested, plus §4 (the two human-annotation gates, which are easy to
conflate) and §5 (who does what). Everything here is diagnosis and pointer material; **no fixes
were attempted.**

> ### ⚠ Read this first: there are TWO different human-annotation tasks
>
> | | **Task A — judge validation** | **Task B — corpus defect** |
> |---|---|---|
> | Artifact | `results/ex4_free_response/validation_sheet_pilot_bidir.csv` | `results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv` |
> | Size | **74 rows** | **33 SciQ items** |
> | Question asked | *Does the model answer mean the same as the gold answer?* | *Is this passage about this question?* |
> | Validates | the **LLM judge** (Qwen3-4B) in Experiment 4 | the **SciQ corpus** |
> | κ computed | human vs LLM judge — **gate at κ ≥ 0.70** | annotator vs annotator on 15 of 33 — **no gate** |
> | Blocks | ✅ cleared 2026-09-08, κ = 0.754 (§4.0); the full run is still held by RUN_QWEN2.5 Entry 3 | nothing; it publishes a mismatch rate |
> | Agent may label? | **No — never** (see §4.2) | Not any more — the prior hand-check is now annotator 1 (§2.5) |
>
> They are unrelated. Two sheets, two κ computations, two purposes. Details: §4 (Task A), §2 (Task B).
>
> **Both sheets are built and waiting.** The rubric for both is
> [`docs/guides/ANNOTATION_GUIDE.md`](guides/ANNOTATION_GUIDE.md); it supersedes the scattered guidance below for
> anything to do with *how to label*.

> All paths are post-reorganization. If a path here does not exist, you are on a pre-cleanup
> checkout — see `PROJECT_READING_GUIDE.md` §5 for the old→new map.

---

## 1. Counterfactual experiment on OpenBookQA

### 1.1 What actually happened — it did not error

**There is no crash, no exception, no aborted run.** All three ex2 runs completed cleanly. Greping
the OBQA logs for `error|traceback|exception|abort|fail` returns only path strings containing the
word, never a failure.

| Artifact | Status |
|---|---|
| [`results/ex2_counterfactual/counterfactual_obqa_dryrun.log`](../results/ex2_counterfactual/counterfactual_obqa_dryrun.log) | clean, generation-only |
| [`results/ex2_counterfactual/counterfactual_obqa_test_full.log`](../results/ex2_counterfactual/counterfactual_obqa_test_full.log) | clean, 243.2 s total |
| [`results/ex2_counterfactual/counterfactual_obqa_test_exact_tier.log`](../results/ex2_counterfactual/counterfactual_obqa_test_exact_tier.log) | clean, 244.4 s total |

**The failure is eligibility collapse plus target-set collapse, not an error.** The single line that
shows it, present in both the dry-run and full logs:

```
Generated 145 usable counterfactual passages out of 500 samples
  (tiers: {'exact': 42, 'glued': 1, 'morphological': 5, 'partial': 97, 'none': 355})
```

versus SciQ in [`counterfactual_sciq_test_full.log`](../results/ex2_counterfactual/counterfactual_sciq_test_full.log):

```
Generated 860 usable counterfactual passages out of 884 samples
  (tiers: {'exact': 813, 'glued': 0, 'morphological': 11, 'partial': 36, 'none': 24})
```

### 1.2 The degradation, quantified

Read from `summary.counterfactual_generation`, `summary.per_model_metrics[primary_model]` and
`summary.adjusted_kda` in the three result files. Primary model is `Riiid/kda-mpnet-base-race`
throughout.

| | SciQ `partial` | OBQA `partial` | OBQA `exact` |
|---|---:|---:|---:|
| Samples | 884 | 500 | 500 |
| **Counterfactual-eligible** | **860 (97.3%)** | **145 (29.0%)** | **42 (8.4%)** |
| `none` tier (no substitution possible) | 24 (2.7%) | 355 (71.0%) | — |
| Share of eligible resting on the loosest `partial` tier | 36/860 (4.2%) | **97/145 (66.9%)** | 0 |
| **`both_correct` target set** (what Setting C actually classifies) | **381** | **41** | **16** |
| → `context_dependent` | 287 (75.3%) | 22 (53.7%) | 7 (43.8%) |
| → `prior_dependent` | 85 (22.3%) | 17 (41.5%) | **8 (50.0%)** |
| → `unstable_other` | 9 (2.4%) | 2 (4.9%) | 1 (6.2%) |
| `kda_original_eligible` mean | 0.4767 | 0.3047 | 0.3812 |
| `kda_adjusted_hard` mean | 0.3357 | 0.1275 | 0.1902 |
| `retention_ratio_hard` | 0.7043 | **0.4185** | 0.4991 |
| `context_verified_accuracy_with_fact` | 0.8581 | **0.3517** | 0.4286 |
| `prior_inflation_of_accuracy_with_fact` | 0.0395 | **0.1655** | 0.0476 |

**The operative number is the `both_correct` target set: 381 on SciQ, 41 on OBQA `partial`, 16 on
OBQA `exact`.** Every headline Setting-C claim is a proportion of that set. At n=16 the `exact`-tier
split is 7 vs 8 vs 1 — no interesting hypothesis is separable at that resolution.

**Quantity is not the only problem — the surviving 145 are also low quality.** Standing limitation
**L1** ([`counterfactual_experiment_methodology.md`](ex2_counterfactual/counterfactual_experiment_methodology.md) §42-46):
the pipeline performs *no grammaticality check and no "does some option still hold" check*, so a
`context_dependent` / `prior_dependent` label can rest on a perturbed passage that is not well-formed
English. The stated impact is **3.6% of SciQ `context_dependent` labels but 70.8% of OBQA's**.
[`counterfactual_obqa_analysis.md`](ex2_counterfactual/counterfactual_obqa_analysis.md) §286 adds: *"67% of the eligible
pool has degraded substitutions (`partial` tier, ~81% ungrammatical)"*.

So the OBQA failure is **two compounding problems**, and a fix must address both:

1. **Coverage** — 355/500 produce no counterfactual at all (§1.3).
2. **Quality** — of the 145 that do, ~2/3 are `partial`-tier and ~81% of those are ungrammatical, so
   even the surviving labels are weakly grounded.

`counterfactual_obqa_analysis.md` §144-150 also notes a third, independent reason the intervention
underperforms here: **Setting B was already weak on OBQA** — the gold fact buys only +0.094 accuracy
(0.434 → 0.528) versus +0.383 on SciQ (0.514 → 0.897). *A perturbation cannot remove a dependency
that was never established.* That is not fixable by improving the generator.

### 1.3 Root cause — best hypothesis, with measurement

**Hypothesis: the perturbation is a lexical find-and-replace of the gold answer inside the passage,
and on OpenBookQA the gold answer is almost never in the passage to begin with.**

Measured directly over the two prepared test splits:

| | SciQ (884) | OBQA (500) |
|---|---:|---:|
| Passage words, mean / median | 77.9 / 58 | **9.0 / 8** |
| Gold-answer words, mean | 1.48 | 3.26 |
| **Gold answer appears verbatim in its own passage** | **820/884 = 92.8%** | **47/500 = 9.4%** |

That 92.8% → 9.4% collapse is the whole effect, and it is structural, not a bug:

* SciQ `support` passages are **extractive** — retrieved prose that contains the answer string.
* OBQA `passage` is `fact1`, a **one-clause premise** from which the answer must be *deduced*. The
  fact supports the answer without naming it.

Three OBQA items from `summary.ineligibility_join` in
[`results/ex5_failure_audit/bucket_diagnostics.json`](../results/ex5_failure_audit/bucket_diagnostics.json)
make the mechanism concrete:

| id | question | gold | passage (`fact1`) |
|---|---|---|---|
| 2 | "Predators eat" | `bunnies` | "predators eat prey" |
| 6 | "an electric car contains a motor that runs on" | `ions` | "an electric car contains an electric motor" |
| 1 | "There is most likely going to be fog around:" | `a marsh` | "fog is formed by water vapor condensing in the air" |

In each case there is no gold-answer token in the passage, so `build_counterfactual` has nothing to
rewrite and returns tier `none`.

**Ruled out — answer length is not the discriminator.** The intuitive explanation ("OBQA answers are
3.26 words, too long to match") does not survive measurement. Among the 355 ineligible items the
mean gold-answer length is **3.02** words; among the 145 eligible it is **3.87** — the eligible
answers are *longer*. Multi-word share is 66.8% ineligible vs 73.1% eligible. Do not spend time on a
fix aimed at answer length.

**Prior art in the repo, which already reached the same conclusion at lower resolution:**
[`docs/ex2_counterfactual/counterfactual_obqa_analysis.md`](ex2_counterfactual/counterfactual_obqa_analysis.md) and README §6.6
("OpenBookQA's one-clause deductive facts do not support the same lexical rewrite at scale;
extending the intervention there is open work"). What is new here is the 92.8% → 9.4% verbatim-
containment measurement and the elimination of the answer-length explanation.

### 1.4 Where to start

* Generator: [`code/ex2_counterfactual/counterfactual_passage.py`](../code/ex2_counterfactual/counterfactual_passage.py)
  — 257 lines, no ML. `TIERS`, `build_counterfactual`, `morphological_variants`.
* Tier ladder is `exact` → `glued` → `morphological` → `partial` → `none`; the tier used is recorded
  per sample as `substitution_tier`.
* Regenerate the eligibility picture in ~2 seconds, no models loaded:

```bash
.venv/bin/python code/ex2_counterfactual/run_counterfactual_experiment.py \
  --data datasets/openbookqa/obqa_test_full.json --dry-run \
  --out /tmp/obqa_dry.json --log-file /tmp/obqa_dry.log
```

* Per-sample generation records, including every `none`-tier item:
  [`results/ex2_counterfactual/counterfactual_passages_preview_obqa.json`](../results/ex2_counterfactual/counterfactual_passages_preview_obqa.json)
  (`generation` block + `samples` array).
* **Read `counterfactual_experiment_methodology.md` §9 before reusing any Setting-C label** — L1
  says so explicitly for exactly these OBQA labels.
* **Caution:** any change to `counterfactual_passage.py` changes SciQ's numbers too. The SciQ
  generation stats are currently reproducible exactly (884 samples, 860 eligible, `mean_substitutions`
  1.8941860465116278) — verify that still holds after any edit, or state that SciQ was intentionally
  re-baselined.

---

## 2. The 33-question manual labeling set — ✅ COMPLETE 2026-09-09

> ### Result
>
> | | |
> |---|---:|
> | Confirmed SciQ topic mismatches | **12 / 884 = 1.36%** (was 13 / 884 = 1.47%, single-annotator) |
> | Double-annotation n | **15** |
> | Observed agreement | **0.933** |
> | **Cohen's κ** | **0.857** |
> | Disagreements | **1** — item #584, adjudicated `ON_TOPIC` |
> | Gate | none (κ ≥ 0.70 is Task A's alone) |
>
> Annotator 1 = the prior LLM hand-check; annotator 2 = a human, working blind from
> `corpus_defect_sheet_sciq33_BLIND.csv` and locked before comparison. **One rater is an LLM
> agent**, so this is human-vs-LLM κ — but it is the project's first inter-annotator κ of any kind.
> Full write-up: [`corpus_defect_audit.md`](ex5_failure_audit/corpus_defect_audit.md) **§8**. Artifacts:
> `corpus_defect_kappa.json`, `corpus_defect_adjudication.json`,
> `corpus_defect_sheet_sciq33_BLIND.LOCK.json`.
>
> **Notable:** the 12 items that entered via the Setting-C-ineligibility screen alone had never been
> hand-checked by anyone; the human census found **zero** mismatches among them (§8.5). And the sole
> overturned item was annotator 1's — the LLM rater — which is the case the double annotation
> existed to catch (§8.3).
>
> The subsections below are the pre-execution scoping, retained as the record of how the set was
> defined.

It exists and is fully specified. It is **not** a subset of the 408 flagged pool, and it is unrelated
to the 328/80/20/145/42 counts.

### 2.1 Where it is defined

| | |
|---|---|
| Primary spec | [`docs/ex5_failure_audit/corpus_defect_audit.md`](ex5_failure_audit/corpus_defect_audit.md) **§6**, lines 243 and 267–276 |
| Superseded plan it replaces | [`docs/ex5_failure_audit/unexploited_buckets_analysis.md`](ex5_failure_audit/unexploited_buckets_analysis.md) §6, line 449 — "reduced from 120 items / 6–7 h to 33 items / ~1.5 h" |
| Screen data | [`results/ex5_failure_audit/passage_overlap_audit.json`](../results/ex5_failure_audit/passage_overlap_audit.json) · [`results/ex2_counterfactual/results_counterfactual_sciq_test_full.json`](../results/ex2_counterfactual/results_counterfactual_sciq_test_full.json) |
| Existing partial hand-labels | [`results/ex5_failure_audit/passage_overlap_handcheck.json`](../results/ex5_failure_audit/passage_overlap_handcheck.json) |

### 2.2 What the set is

**SciQ test split only.** The union of two screens:

1. **Question/passage overlap < 0.40** — 21 items. Metric: share of question content words present
   in the passage, stopworded, with singular/plural stemming.
2. **Setting-C-ineligible** — 24 items, i.e. `counterfactual_valid == false`, meaning the passage
   never lexically states the gold answer.

Union = **33 unique items**, which the audit reports as capturing **13 of 13** known topic
mismatches. Established rates from that audit, over all 1,384 test items with no sampling:
**SciQ 13/884 = 1.47%** genuinely off-topic passages (*single-annotator; superseded by the
adjudicated **12/884 = 1.36%**, `corpus_defect_audit.md` §8*); **OBQA 0%** in the checked sample — OBQA's low
lexical overlap is the deductive nature of `fact1`, not a defect (the same structural fact that
drives §1.3). Screen breakdown (21 + 24 − 12 = 33):

* in both screens: **12** items
* overlap-screen only: **9** items
* ineligibility-screen only: **12** items

### 2.3 The exact 33 IDs

Reproduced from committed data this session (`id` == index into `datasets/sciq/sciq_test_full.json`):

```
23, 33, 54, 127, 153, 183, 201, 203, 237, 251, 306, 321, 381, 386, 412, 525, 566,
584, 594, 598, 602, 610, 619, 639, 655, 739, 746, 754, 760, 762, 828, 843, 844
```

Regenerate them:

```bash
.venv/bin/python -c "
import json
po=json.load(open('results/ex5_failure_audit/passage_overlap_audit.json'))['datasets']['sciq']
tail={int(k) for k,v in po['all_overlaps'].items() if v < 0.40}
cf=json.load(open('results/ex2_counterfactual/results_counterfactual_sciq_test_full.json'))
inel={r['id'] for r in cf['results'] if not r.get('counterfactual_valid')}
print(sorted(tail|inel), len(tail|inel))
"
```

> ⚠️ **Do not read the 33 out of `passage_overlap_audit.json`'s `tail_items` array.** That export was
> written at `tail_threshold: 0.1` and contains only **12** items. The §6 plan's arithmetic uses
> **< 0.40**, which is only recoverable from the `all_overlaps` dict (884 entries, id→overlap). The
> snippet above uses the correct field.

### 2.4 The intended labeling scheme

Per `corpus_defect_audit.md` §6, a four-step plan, ~1.5 hours total:

| Step | Scope | Effort |
|---|---|---|
| 1. Census the SciQ union screen | 33 items | ~45 min |
| 2. Double-annotate 15 of the 33, report Cohen's κ | 15 items | ~20 min |
| 3. Adjudicate and publish a **corpus mismatch rate** for SciQ | — | ~30 min |
| 4. OpenBookQA | **nothing** — 0/25 defects found, with a structural explanation | — |

**The judgement being made is binary and deliberately narrow:** *is this passage about this
question?* The document argues the κ gate matters more here than for the five-way failure taxonomy,
precisely because the question is more objective — "if κ is not high, the rubric rather than the
sample is at fault."

**Explicitly dropped from the plan, with reasoning:** mis-key rate estimation. Across 55 items read
in the prior round there were 0 unambiguous mis-keys and 1 contested key (OBQA #228, `insects` vs
`invertebrates`). At a true rate near 2%, ±5 pp needs n≈30 but a ±5 pp interval around 2% is
compatible with 0%, so the estimate cannot distinguish "rare" from "absent".

**Also explicitly not recommended:** re-annotating `both_wrong` as a bucket. 87.5% of OBQA
`both_wrong` items were established as model-side failures, not item defects.

> **How to actually make the judgement** is not in §6 and not here — it is in
> [`docs/guides/ANNOTATION_GUIDE.md`](guides/ANNOTATION_GUIDE.md) §2, which extracts the binary definition from
> `corpus_defect_audit.md:139` and §6, adds a decision procedure for the topically-adjacent cases,
> and carries synthetic practice items. §6 supplies the *what*; the guide supplies the *how*.
> Note that despite §6's wording there is **no gate** on this κ — see §2.5.

### 2.5 The three open questions — ALL CLOSED 2026-09-08

The first draft of this handoff listed four unknowns here. All are now resolved by decision, and
recorded in three places so they do not need re-deciding:
[`docs/guides/ANNOTATION_GUIDE.md`](guides/ANNOTATION_GUIDE.md) §2.6 (the reasoning),
`results/ex5_failure_audit/corpus_defect_comparison_subset.json` → `_provenance` (machine-readable),
and this section.

| Was open | Now |
|---|---|
| No sheet exists for the 33 | **Built.** `results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv`, 33 rows, `verdict` column empty, generated by `code/ex5_failure_audit/build_corpus_defect_sheet.py` |
| Whether `passage_overlap_handcheck.json` partially satisfies step 1 | **Repurposed as annotator 1** — it is not step 1, it is the *first* of the two annotations |
| Who the second annotator is | **The human.** The prior hand-check is annotator 1; the human filling the sheet is annotator 2 |
| Whether an agent may do a first pass | **Moot, and now excluded.** The first pass already exists. A new agent pass would displace annotator 1, not add one |

**Which 15.** `random.Random(20260904).sample(sorted(prior_labelled_ids), 15)` over the 21 items
annotator 1 categorised — not a random 15 across all 33, because only those 21 have a first verdict
to compare against. The result:

```
23, 153, 201, 203, 251, 306, 321, 412, 584, 598, 602, 619, 655, 746, 828
```

The remaining 18 (6 unselected from the 21, plus the 12 that entered via the Setting-C-ineligibility
screen alone) are labelled single-pass. They count toward the census and the published mismatch
rate but not toward κ.

**Blind: yes, one-directional.** Annotator 1's labels were fixed in an earlier session, so the whole
burden falls on annotator 2, who must not read `corpus_defect_audit.md` §3.1–3.2 or
`passage_overlap_handcheck.json` until the sheet is locked. The scorer reads annotator 1 straight
out of the committed artifact, so no second file carrying those verdicts is ever created. The sheet
repeats the warning in its own `#` header.

**No gate.** `KAPPA_GATE = 0.70` is pre-registered for Task A only. Per
`corpus_defect_audit.md:278`, a low κ here indicts the rubric rather than the sample.

**Standing limitations that survive the decision**, and belong beside the published number:

* **One rater is an LLM agent.** This is human-vs-LLM agreement, not human-vs-human. It is still the
  project's first two-annotator κ of any kind — every other hand label remains single-pass.
* **n = 15**, so κ moves several hundredths per disagreement. Always quote n beside it.
* **κ covers the low-overlap items only**, since that is the set annotator 1 read. The 12
  ineligibility-only items get no second read; the mismatch rate rests on all 33, κ on 15.

### 2.6 Tooling for Task B

| Script | Purpose |
|---|---|
| [`code/ex5_failure_audit/build_corpus_defect_sheet.py`](../code/ex5_failure_audit/build_corpus_defect_sheet.py) | Re-derives the 33 from committed data, draws the pre-registered 15, writes the sheet + manifest. Refuses to overwrite a sheet carrying labels unless `--force`. |
| [`code/ex5_failure_audit/kappa_corpus_defect.py`](../code/ex5_failure_audit/kappa_corpus_defect.py) | Cohen's κ, annotator 1 (hand-check) vs annotator 2 (sheet), over the 15. Reuses `compute_kappa.py:42-66` with `LABELS` swapped; applies no gate. |

```bash
# after the sheet is locked
.venv/bin/python code/ex5_failure_audit/kappa_corpus_defect.py \
    --out results/ex5_failure_audit/corpus_defect_kappa.json
```

Both follow the ex5 convention — `utils.paths.resolve` / `ensure_parent`, `--out`, a printed
report, no `--log-file` (that is an ex1/ex2 model-running convention; no ex5 audit script has one).

---

## 3. RQ2 — formulation status and open directions

### 3.1 Specification

README **§2.4** (lines 164–225) defines four estimators plus a derived accuracy; **§2.5** (lines
217–235) defines the three-setting design and the three-way `both_correct` classification.

### 3.2 Implemented and validated

All four estimators are implemented in
[`code/ex2_counterfactual/run_counterfactual_experiment.py`](../code/ex2_counterfactual/run_counterfactual_experiment.py)
and computed in every run:

| Estimator | § 2.4 form | Code | Reported as |
|---|---|---|---|
| $KDA_{adj}^{hard}$ | indicator gate on argmax | lines 377, 538, 573 | `kda_adjusted_hard` |
| $KDA_{adj}^{soft}$ | probability-mass weight | lines 378, 539, 574 | `kda_adjusted_soft` |
| $KDA_{adj}^{excl}$ | question-level filter over $\mathcal{Q}_{ctx}$ | `_exclusion_variant`, `compute_adjusted_kda` | `kda_adjusted_sample_exclusion` (**both readings since 2026-09-09**) |
| $\text{Acc}^{verified}_{wf}$ | context-verified accuracy | lines 496–502 | `context_verified_accuracy_with_fact` |

Supporting outputs: `retention_ratio_{hard,soft,sample_exclusion}`,
`prior_inflation_of_accuracy_with_fact`, and per-sample `kda_adjusted_hard` / `kda_adjusted_soft`.

`Q_ctx` is implemented as: drop a question iff its **ensemble** record has
`baseline_category == "both_correct"` **and** `counterfactual_class == "prior_dependent"`.

**RESOLVED 2026-09-09.** Both readings are now computed and reported —
`mean_over_retained` (README §2.4's $\mathbb{E}_{q \in \mathcal{Q}_{ctx}}$, denominator `n_kept`)
and `mean_over_eligible_zero_filled` (the historical figure, denominator `n_eligible`) — together
with a nine-way $\mathcal{Q}_{ctx}$ sensitivity sweep and the identity
`zero_filled = retained × keep_rate`, asserted every run. Corrected figures for the three committed
runs: [`results/ex2_counterfactual/adjusted_kda_corrected.json`](../results/ex2_counterfactual/adjusted_kda_corrected.json),
regenerated with no GPU by
[`code/ex2_counterfactual/recompute_adjusted_kda.py`](../code/ex2_counterfactual/recompute_adjusted_kda.py)
(which verifies every pre-existing figure re-derives exactly before writing).

**Validated on:** SciQ test (884) and OpenBookQA test (500, twice — `partial` and `exact` tiers),
with the `KDA_small` 4-model encoder ensemble. Numbers in §1.2 above. Methodology walkthrough:
[`docs/ex2_counterfactual/counterfactual_experiment_methodology.md`](ex2_counterfactual/counterfactual_experiment_methodology.md).

### 3.3 Open, unexplored, or specified-but-unvalidated

Derived from the repo as it stands. Not recommendations — just what is genuinely unfinished.

1. ~~**Setting C has never been run with an LLM solver.**~~ **CLOSED 2026-09-09 — see
   [`docs/ex2_counterfactual/e1_counterfactual_llm_scale.md`](ex2_counterfactual/e1_counterfactual_llm_scale.md).** Qwen3-4B scored all
   three settings on SciQ; the `both_correct` target set is **819** (2.15× the encoders' 381).
   **The intervention survives saturation**: 62.5% `prior_dependent` [95% CI 59.2–65.8], 37.1%
   `context_dependent`, 0.4% unstable. Prior inflation of Acc$_{wf}$ is **+59.9 pp** against the
   encoder ensemble's +4.0 pp. Both correctness gates passed (Setting A/B bit-exact vs ex1b;
   884/884 passages identical to the committed preview).

   **Two results that were not anticipated.** (a) Cross-solver agreement on the Setting-C class is
   **κ = 0.046** — the label is a property of the solver, not the question, which undercuts RQ3's
   filtering premise (§3 of that doc). **This remains open and is now the largest RQ2/RQ3 risk.**
   (b) At LLM scale `prior_dependent` conflates "ignored the context" with "correctly detected the
   context is false", so 62.5% is an upper bound — **tested and largely ruled out by E2 below.**

1b. **E2 — prior-dependence vs rejection of a false context. CLOSED 2026-09-09**, see
   [`docs/ex2_counterfactual/e2_prior_vs_rejection.md`](ex2_counterfactual/e2_prior_vs_rejection.md). Two independent probes, both
   rejecting the rejection hypothesis: under an explicit order to treat the passage as
   authoritative only **26.4%** [22.7, 30.4] of `prior_dependent` items flip, and the model's own
   plausibility penalty does not predict which items refuse (**AUC 0.496** on the decisive
   within-class test; 0.533 across classes). The control confirms the instruction is not merely
   disruptive — gold-passage accuracy moves by −0.0011. **Genuine prior-dependence is bracketed at
   46.0%–62.5%**, against the encoder ensemble's 22.3%. Residual open question: the 26.4% who do
   flip are not the implausible ones either, so prompt sensitivity is the unexplained remainder,
   and whether the context-priority instruction belongs in the metric's *definition* is now a live
   design choice.

2. ~~**`Q_ctx` is defined only on the ensemble vote.**~~ **CLOSED 2026-09-09.** Nine definitions are
   now implemented (`q_ctx_definitions`) and swept in every run: ensemble vote, ensemble
   all-buckets, each solver individually, and any/majority/all-member votes. The answer is that the
   choice **does not matter** for the score — across all three committed runs the retained mean
   stays within 93.9%–101.7% of baseline while the drop count ranges from 0 to 297 of 860. It
   matters only for the zero-filled form, which it rescales (64.5%–100% on SciQ).

3. ~~**The exclusion variant credits dropped items 0 rather than excluding them.**~~ **CLOSED
   2026-09-09** — and the resolution changed a headline number. Both readings are now reported
   (§3.2). They differ by exactly the keep rate. The substantive finding: under README §2.4's
   reading the estimator is **inert** — SciQ 0.4781 against a 0.4767 baseline, i.e. 100.3%
   retention, versus the 90.4% previously published. Dropping prior-dependent questions does not
   lower the mean because they are not low-`KDA_cont` questions (dropped 0.4640 vs kept 0.4781; AUC
   0.536, and 0.323 *inverted* on OBQA `partial`). `KDA_cont` carries almost no information about
   prior-dependence, which is the RQ1 thesis reappearing inside the estimator built to repair it.
   README §2.4 and §6.3, `counterfactual_experiment_methodology.md` §6 and
   `counterfactual_obqa_analysis.md` §2.3 are all updated.

4. ~~**`unstable_other` is credited to neither class** and is never modelled.~~ **CLOSED
   2026-09-09.** The estimators were in fact *inconsistent*, which no document had recorded:
   hard, soft and context-verified accuracy scored `unstable_other` as a failure while sample
   exclusion retained it at full `KDA_cont`. Three conventions (strict / lenient / abstain)
   were computed on all three runs; the movement is up to **29 pp** of hard retention on OBQA.
   **Strict** was adopted for all four estimators — `abstain` would reintroduce RQ1's
   silently-shrinking denominator, and `lenient` credits an incoherent perturbation as
   knowledge dependency and halves the SciQ-vs-OBQA gap. Decision and numbers:
   [`docs/ex2_counterfactual/unstable_other_convention.md`](ex2_counterfactual/unstable_other_convention.md) ·
   [`results/ex2_counterfactual/unstable_other_conventions.json`](../results/ex2_counterfactual/unstable_other_conventions.json).
   Residual open question: an `unstable_other` item still cannot be separated into "solver
   broke" vs "perturbation was word salad".

5. **No adjusted $KDA_{disc}$.** §2.4 defines adjusted forms of $KDA_{cont}$ only. Since ex1b showed
   $KDA_{disc}$'s denominator collapsing to 41 of 884 questions, whether the counterfactual gate
   repairs or worsens the discrete variant is unasked.

6. **The intervention is single-target.** Exactly one distractor is chosen as the counterfactual
   target per question. Sensitivity to *which* distractor, or averaging over all three, is neither
   implemented nor measured.

7. **No threshold/operating-point analysis.** README §6.5 mentions the original paper's 0.8 operating
   point for a filtering arm, but no ROC, no threshold sweep, and no filter/rerank evaluation exists
   for any of the three adjusted estimators.

8. **OBQA remains unresolved as a second dataset** — see §1. Until the generator handles deductive
   one-clause facts, every cross-dataset RQ2 claim rests on n=41 (`partial`) or n=16 (`exact`).

9. **No third dataset.** Only SciQ and OpenBookQA are prepared. Both preparers are in
   `code/pre_data/`; adding a corpus with extractive passages would test whether the SciQ result
   generalises or is a property of extractive support text.

10. **RQ3 has no code at all.** README §6.5 specifies three arms (No Filtering / KDA Filtering / Our
    Disentangled Filtering) with QG-SMS-based validation. No package exists, and **no ordinal is
    reserved for it** — it takes the next free experiment number when it starts. The reservation
    drifted from `ex6_` to `ex7_` to `ex8_` as those ordinals were each claimed by work that got
    built first, so pre-assigning one has been abandoned.

### 3.4 Fastest way to re-derive any number above

```bash
.venv/bin/python -c "
import json
d=json.load(open('results/ex2_counterfactual/results_counterfactual_sciq_test_full.json'))['summary']
print(json.dumps(d['adjusted_kda'], indent=2)[:1500])
print(json.dumps(d['per_model_metrics'][d['primary_model']], indent=2)[:800])
"
```

Swap the filename for `results_counterfactual_obqa_test_full.json` or
`results_counterfactual_obqa_test_exact_tier.json`.

---

## 4. The free-response human-validation gate (Task A) — ✅ CLEARED 2026-09-08

This is the item that blocks the largest piece of downstream work. It is **not** the 33-item task in §2.

### 4.0 Result — the gate passed

**Labelled 2026-09-08, all 74 rows, by the human annotator, blind to the judge's columns.**

| | |
|---|---|
| κ, forward-only verdict | **0.614** — *below* the 0.70 gate |
| κ, both-directions-agree verdict | **0.754** — **passes** |
| `best_kappa` / `gate_passed` | 0.754 / `true` |
| Raw agreement | 0.811 forward · 0.892 both-directions |
| Report | [`results/ex4_free_response/kappa_report.json`](../results/ex4_free_response/kappa_report.json) |

**The gate passes on one mechanism only.** The forward-only verdict — the one that would be quoted
by default — scores 0.614 and fails. Only the both-directions-agree mechanism, which scores an item
CORRECT only when the judge said CORRECT in both directions, clears 0.70. Since `compute_kappa.py`
takes the max of the two (line 151-157), the gate registers as passed, but the licensed headline is
specifically the both-directions band, not the forward verdict.

**The stratum split is the substantive finding**, and it is exactly the case `compute_kappa.py`'s
docstring (lines 13-16) anticipated:

| Stratum | n | Agreement | κ |
|---|---:|---:|---:|
| `agree` (judge self-consistent) | 56 | 0.911 | **0.816** |
| `flip` (judge unstable under swap) | 18 | 0.500 | **0.069** |

Where the judge is self-consistent it is reliable; where it flips it is worth nothing — κ = 0.069 is
chance agreement. That is a usable result rather than a failure, and it is the mechanism behind the
headline: discarding flipped items is precisely what the both-directions verdict does.

**Direction of the disagreement.** On the forward verdict, human=INCORRECT / judge=CORRECT occurs 13
times against a single case the other way. **The judge over-credits**; it does not disagree
symmetrically. 14 of 74 items disagree on forward, 8 of 74 on both-directions.

**Provenance of the labelling.** The annotator worked from a blind copy with the judge's columns
stripped ([`validation_sheet_pilot_bidir_BLIND.csv`](../results/ex4_free_response/validation_sheet_pilot_bidir_BLIND.csv)
+ its manifest, generated by `build_blind_sheet.py`), and the verdicts were merged back on the
derivation-stable `pair_key` by `merge_blind_verdicts.py`, which asserts a 1:1 join and refuses any
write that would touch another column. Rubric: [`docs/guides/ANNOTATION_GUIDE.md`](guides/ANNOTATION_GUIDE.md) §1.

### 4.1 The artifact, as it stands

[`results/ex4_free_response/validation_sheet_pilot_bidir.csv`](../results/ex4_free_response/validation_sheet_pilot_bidir.csv)

| | |
|---|---|
| Data rows | **74** (file is 83 lines: 8 comment lines + header + 74 rows) |
| Column to fill | **`human_verdict`** — ✅ filled on all 74 rows, 2026-09-08 |
| Accepted values | `CORRECT` / `INCORRECT`. `compute_kappa.py:78-81` also accepts `C`/`I`, `1`/`0`, `Y`/`N`, `YES`/`NO`, `TRUE`/`FALSE` |
| Other columns | `row_id, dataset, cell, item_id, question, gold_answer, model_answer, judge_forward, judge_reverse, judge_agree, stratum` |
| Composition | dataset: obqa 48 / sciq 26 · cell: A′ 44 / B′ 30 · 7 strata over dataset × cell × {agree, flip} |
| Generated by | `build_validation_sample.py --tag pilot_bidir` |

**Why 74 rows and not 150.** `build_validation_sample.py:148` defaults to `--n 150`. Only 74 pilot
items had been judged, so the stratified sampler returned every available item. **This is not a bug**
— it is the consequence of the pilot being n=25 per dataset per cell. Re-running the builder after a
full-scale run is what makes the stratified random sampling meaningful.

**The sheet's own instructions** (comment preamble, lines 1–8 of the CSV) are worth following:

> Fill ONLY the `human_verdict` column with CORRECT or INCORRECT. CORRECT means: the model_answer
> means the same thing as the gold_answer, in the context of the question. Ignore spelling,
> capitalisation and phrasing. A more specific or more general answer counts as CORRECT only if it
> identifies the same thing. Please do NOT read the judge_forward / judge_reverse columns before
> deciding — they are included so the sheet is self-contained for scoring, not as a prompt. Hide
> them if you can.

### 4.2 Why an agent must NOT fill this in

**The purpose of this sheet is to validate the LLM judge.** The judge is Qwen3-4B; it grades
free-response answers that Qwen3-4B itself generated, and the pilot found it flips **24.3%** of its
verdicts when reference and student are swapped. Cohen's κ here measures agreement between *the LLM
judge* and *an independent human*.

If an agent — itself an LLM — fills the `human_verdict` column, the measurement stops being an
independence check and becomes one language model agreeing with another. A high κ obtained that way
would be evidence of nothing. This is a student grading their own exam.

**An agent may not fill this column under any framing** — not "to save time", not "as a first pass
for the human to correct", not "to demonstrate the tooling". A pre-filled column anchors the human
annotator and destroys the independence the gate exists to establish. If a future session is asked
to do it, the correct response is to decline and explain why.

### 4.3 The gate, and what happens if it fails

`KAPPA_GATE = 0.70` in [`code/ex4_free_response/compute_kappa.py:38`](../code/ex4_free_response/compute_kappa.py),
marked **"pre-registered; do not change after seeing results"** — declared in code before the human
labels exist, so the decision rule cannot be adjusted retroactively.

```bash
.venv/bin/python code/ex4_free_response/compute_kappa.py \
  --sheet results/ex4_free_response/validation_sheet_pilot_bidir.csv
```

| Outcome | Consequence |
|---|---|
| **κ ≥ 0.70** | The judge is reliable enough. `acc_band_low` / `acc_band_high` may be published as the headline, with κ quoted alongside. |
| **κ < 0.70** | The judge is **NOT** reliable enough. Publish `acc_normalised` — the deterministic floor from matching stages 1–2 only — as the headline, and the band as a diagnostic only. **Do not publish `acc_judged` until the judge is fixed.** |

Two κ values are computed (`compute_kappa.py:108-116`): against the judge's **forward** verdict, and
against its **both-directions-agree** verdict. Agreement is also reported separately inside the
`agree` and `flip` strata — if the judge is reliable where self-consistent and unreliable where it
flips, that is itself a usable result and would justify reporting the band.

The script prints `Nothing to score yet -- fill the human_verdict column first.` on an empty sheet,
so it is safe to run before labeling to check the plumbing.

### 4.4 The second gate — Qwen2.5-7B as an independent judge

[`RUN_QWEN2.5.md`](../RUN_QWEN2.5.md) **Entry 3**, §4, marked **⚠ CRITICAL PATH** — the only one of
the four entries on it.

**Why it is a separate gate from §4.1.** Task A checks the judge against a *human*. Entry 3 checks
it against a *different model*. Both attack the same threat from different angles: the judge is
Qwen3-4B grading Qwen3-4B's own output, so **self-judging bias is a first-order threat**, and neither
a human check nor a cross-model check alone rules it out.

| | |
|---|---|
| Cost | ~2 minutes of GPU time |
| Inputs needed | `results/ex4_free_response/results_free_response_{sciq,obqa}_pilot_bidir.json` uploaded to the notebook |
| Self-contained | Yes — the code block in `RUN_QWEN2.5.md` §4 pastes straight into Colab/Kaggle, imports nothing from this repo |
| Why not local | Qwen2.5-7B (7.62B params) does not fit the 4 GB RTX 3050 with useful headroom |
| Pilot findings it addresses | judge decides 76–100% of items; flips 24.3% under swap; band widths SciQ A′ 0.120, OBQA A′ 0.240, OBQA B′ 0.360 |

The other three entries (1 — KDA saturation, 2 — persona simulation, 4 — free-response generation)
are queued but explicitly **not blocking**.

### 4.5 Sequencing

1. ~~Label the 74 rows by hand (§4.1–4.2) — **human only**~~ ✅ done 2026-09-08
2. ~~Run `compute_kappa.py`, record κ (§4.3)~~ ✅ done — κ = 0.754, gate passed (§4.0)
3. Run `RUN_QWEN2.5.md` Entry 3 on Colab/Kaggle (§4.4) — **human only** ⬜ **still outstanding**
4. **Only once both clear**, launch the full-scale free-response run — **not yet**: step 3 is
   independently required. `RUN_QWEN2.5.md:24` states the full run is blocked on *"(a) the human
   validation gate … and (b) entry 3"*, and
   `docs/ex4_free_response/plan_option_free_response_experiment.md:13` says *"Do not launch the full run until both
   clear."* Task A clearing does **not** on its own release the run.
5. Re-run `build_validation_sample.py --tag full --n 150` afterwards — with a full-scale judged pool,
   stratified random sampling finally does real work and yields the intended 150 rows

`docs/ex4_free_response/plan_option_free_response_experiment.md` states this directly: *"Do not launch the full run
until both clear."*

### 4.6 Update — Entry 3 completed 2026-09-24

`RUN_QWEN2.5.md` Entry 3 (§4.4 above) has now been run: Qwen2.5-7B judged the same 74 residual
items from the Qwen3-4B bidirectional pilot, in both directions. Full detail and the 2×2 cross-tab:
[`docs/ex4_free_response/plan_option_free_response_experiment.md`](ex4_free_response/plan_option_free_response_experiment.md)
§10.1.

| | |
|---|---|
| Raw agreement (Qwen2.5-7B vs Qwen3-4B, forward verdicts) | 77.0% |
| Cohen's κ, forward-only | 0.541 |
| Cohen's κ, both-directions-agree | 0.545 |
| `best_kappa` / gate (`KAPPA_GATE = 0.70`) | **0.545 / gate NOT passed** |

This is a *different* comparison from the one that cleared in §4.0 above: §4.0 measured
human-vs-Qwen3-4B-judge agreement (passed, κ = 0.754); this measures Qwen2.5-7B-vs-Qwen3-4B-judge
agreement (not passed, κ = 0.545). Both apply the same pre-registered `compute_kappa.py` gate to
their own comparison; one clearing does not imply the other would, and it did not here. Entry 3 is
no longer *outstanding* (§4.5 step 3, §5's table below), but the full Qwen3-4B-generated run it
gates remains blocked on the same rule as before — the gate itself was not cleared, only the item
of running it was.

A separate, full-scale (884+500), self-judged Qwen2.5-7B free-response run also exists (Entry 4, run
beyond its specified n=25 pilot) — see
[`plan_option_free_response_experiment.md`](ex4_free_response/plan_option_free_response_experiment.md)
§10.2. It is graded by its own generator and does not depend on Entry 3's gate.

---

## 5. Who does what

| Task | Who | Why | Where |
|---|---|---|---|
| ~~Label 74 rows in `validation_sheet_pilot_bidir.csv`~~ | **Human only** | ✅ **done 2026-09-08**, blind copy + `pair_key` merge | §4.0 |
| ~~Run `compute_kappa.py` on the filled sheet~~ | Agent or human | ✅ **done** — κ = 0.754, gate passed | §4.0 |
| `RUN_QWEN2.5.md` Entry 3 (second judge, Colab/Kaggle) | **Human** | Needs a cloud GPU session the agent cannot open | §4.4 |
| `RUN_QWEN2.5.md` Entries 1, 2, 4 | **Human** | Same; not blocking | `RUN_QWEN2.5.md` §0 |
| Full-scale free-response run, after both gates clear | Agent | Local, scripted. **Still blocked** — gate (a) cleared, gate (b) Entry 3 has not | §4.5 |
| ~~Label the 33 rows~~ | **Human** (annotator 2) | ✅ **done 2026-09-09**, blind copy, locked before comparison | §2 |
| ~~Run `kappa_corpus_defect.py` on the locked sheet~~ | Agent or human | ✅ **done** — κ = 0.857, n = 15 | §2 |
| ~~Adjudicate and publish the SciQ mismatch rate~~ | Human decision, agent can draft | ✅ **done** — #584 → `ON_TOPIC`; 12/884 = 1.36% published in `corpus_defect_audit.md` §8 | §2 |
| OBQA counterfactual **root-cause diagnosis** | ✅ Agent — **done this session** | See §1.3 | §1 |
| OBQA counterfactual **fix** | Agent | Code change; affects SciQ numbers, see §1.4 caution | §1.4 |
| RQ2 open directions (§3.3, items 1–10) | Agent, except item 1 | Item 1 (LLM-solver Setting C) may need cloud GPU depending on model size | §3.3 |

**There is no unresolved technical blocker.** Nothing in this project is stuck on an error an agent
could not solve. The two blocking gates (§4) are blocked on *human judgement* and *cloud GPU access*
respectively — both outside what an agent can supply, neither a defect.

---

## Standing caveats

* Every hand label in this project **was** single-annotator, single-pass, no κ — the three
  `_provenance` blocks in `results/ex5_failure_audit/` and `results/ex5_failure_analysis/` say so
  themselves. The Task B census is the first and so far only exception: **κ = 0.857 at n = 15**
  (`corpus_defect_audit.md` §8). It remains a human-vs-LLM κ, not human-vs-human — declare that
  beside the number. `passage_overlap_handcheck.json`'s own `_provenance` still says "no kappa";
  that is now out of date, and §8 is the authority.
* `datasets/` is gitignored and regenerable from `code/pre_data/`, but **neither preparer pins an
  upstream HuggingFace revision**. All committed results reference dataset rows by index. If
  `allenai/sciq` or `allenai/openbookqa` is ever revised upstream, the 33 IDs in §2.3 and every
  `dropped_ids` list stop pointing at the same questions.
* Log and result-file *contents* embed pre-reorganization paths. They are immutable execution
  records; see `PROJECT_READING_GUIDE.md` §5.5.
