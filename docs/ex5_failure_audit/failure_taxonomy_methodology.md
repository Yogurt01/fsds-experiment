# Failure-Taxonomy Labeling Methodology (RQ1)

**Audience:** a reviewer who did not write the code and needs to verify how the five failure
categories were assigned.
**Companion to:** [`docs/ex5_failure_audit/rq1_test_split_failure_analysis.md`](rq1_test_split_failure_analysis.md)
**Labelled artefact:** `results/ex5_failure_analysis/rq1_flagged_questions.json` (408 records)

---

## 0. The short answer, stated up front

| Question | Answer |
|---|---|
| Manual, rule-based, or LLM-as-judge? | **Manual annotation only.** No LLM was used at any point. |
| How many annotators? | **One**, in one pass. |
| Written rubric or decision tree? | **None exists in this repository.** See §2. |
| Inter-rater agreement measured? | **No.** No second rater, no adjudication protocol, no κ. |
| Is there a script that produced the labels? | **No.** See §1.2. |
| Was any part automated? | Yes — *candidate selection* (§3) and *structural cues* (§5) are computed. The **category label itself is not.** |

The source document states this itself. `docs/ex5_failure_audit/rq1_test_split_failure_analysis.md` §5, limitation 1:

> "**Single annotator, no inter-annotator agreement.** All 408 category labels were assigned by one
> annotator in one pass. No second rater, no adjudication protocol, no κ. The four-way split among
> `common_knowledge` / `parametric_knowledge` / `reasoning_shortcut` / `option_leakage` is the most
> fragile part of the study; the `common_knowledge` ↔ `parametric_knowledge` boundary in particular
> is a judgement about what counts as 'school-level', and a different annotator would move items
> across it."

The JSON schema block says the same thing in one line
(`results/ex5_failure_analysis/rq1_flagged_questions.json`, key `schema.failure_category`):

```json
"failure_category": "primary failure mode (single label, manual annotation)"
```

---

## 1. What is and is not reproducible

### 1.1 Reproducible (code exists)

| Stage | Code | Output |
|---|---|---|
| Setting A/B scoring of the 4-model ensemble | [`code/ex1_reproduce_KDA/run_experiment.py`](../../code/ex1_reproduce_KDA/run_experiment.py), students in [`kda_tiny.py`](../../code/ex1_reproduce_KDA/kda_tiny.py) | `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json` |
| Four-bucket assignment | [`categorize_kda_results.py:129-133`](../../code/ex1_reproduce_KDA/categorize_kda_results.py) | `results/ex1_category_questions/basic_category/*.json` |
| Setting C + `prior_dependent` classification | [`run_counterfactual_experiment.py:145-153`](../../code/ex2_counterfactual/run_counterfactual_experiment.py) | `results/ex2_counterfactual/results_counterfactual_sciq_test_full.json` |

### 1.2 NOT reproducible (no code in the repository)

Searching the entire tree for any script that writes the labelled file returns nothing:

```bash
grep -rn "rq1_flagged\|rq1_failure" --include="*.py" .   # → no matches
```

There is **no committed script** that (a) applies the three detection criteria to build the
flagged pool, (b) computes the `structural_cues`, or (c) writes
`rq1_flagged_questions.json`. The file is a data artefact whose generating process is not in
version control. A reviewer can verify the *contents* of the file against the upstream results
files by hand, but cannot re-run the selection.

**Update since this document was first written.** The generating script is still absent, but the
logic it encoded has been fully recovered by reimplementation and verified exact against the
artefact — see [`provenance_rq1_flagged_questions.md`](provenance_rq1_flagged_questions.md). The
selection in §3 is now re-derivable with
[`code/ex5_failure_audit/rebuild_flagged_pool.py`](../../code/ex5_failure_audit/rebuild_flagged_pool.py); the manual
labels in §4 remain irreproducible by their nature.

---

## 2. The rubric question — read this carefully

There is **no rubric document, decision tree, or annotation guideline file** anywhere in the
repository. What exists is:

1. the five category **names**;
2. one or two **worked examples per category** with a prose justification, in
   `docs/ex5_failure_audit/rq1_test_split_failure_analysis.md` §3.1–§3.5;
3. a one-line gloss of each in the executive summary and §4.4.

The operational definitions below are **reconstructed by the present author from those worked
examples**. They were not written down before annotation and were not available to the annotator
as a checklist. They should be read as *a description of what the labels appear to mean*, not as
the instrument that produced them. Any reviewer re-annotating from these definitions is running a
new experiment, not reproducing the old one.

---

## 3. Stage 1 — candidate selection (automated, reproducible)

A question enters the flagged pool if it meets **any** of the criteria below. These are stated in
`rq1_test_split_failure_analysis.md` §1.

| Flag | Criterion as written | SciQ | OBQA |
|---|---|---:|---:|
| `C1_kda_cont` | `KDA_cont ≥ 0.70` **and** primary model correct in Setting A | 15 | 0 |
| `C1_kda_disc` | discrete-KDA proxy `= 1.0` **and** primary model correct in Setting A | 167 | 13 |
| `C2_both_correct_conf` | `both_correct` bucket **and** `P(R^q=1) ≥ 0.70` (primary model) | 140 | 76 |
| `C3_prior_dependent` | Setting-C class `prior_dependent` | 115 | — |
| | **Union** | **328 (37.1%)** | **80 (16.0%)** |

Three documented deviations from the literal brief (§1 of the source doc):

- **`KDA_disc` is a proxy.** No human-student annotations exist in the repository, so
  `KDA_disc_proxy(q) = Σ_m 1[¬correct_wof]·1[correct_wf] / Σ_m 1[¬correct_wof]`, i.e. the
  model-ensemble analogue. `= 1.0` means every student that failed without the fact succeeded with it.
- **`P(R^q=1) ≥ 0.70` is read on the primary model, not the ensemble.** On the ensemble average
  the threshold is unreachable — 0/884 SciQ questions clear it, because
  `kda-albert-xlarge-v2-race` is near-uniform and drags the mean to ≈0.33. On
  `Riiid/kda-mpnet-base-race`, 145 SciQ / 81 OBQA questions clear it. This makes the C2 pool
  model-specific.
- **C3 was unavailable for OpenBookQA at the time of writing.** The RQ1 doc states
  "No counterfactual (Setting C) run exists for OpenBookQA." **This is now out of date** — see §6.

### 3.1 Verified counts

The `detection_flags` field is a list, so items appear in multiple rows. Recomputed directly from
the artefact:

**SciQ (328 flagged)** — flag combinations actually present:

| Combination | n |
|---|---:|
| `C1_kda_disc` alone | 81 |
| `C3_prior_dependent` alone | 79 |
| `C2_both_correct_conf` alone | 73 |
| `C1_kda_disc` + `C2_both_correct_conf` | 46 |
| `C1_kda_disc` + `C3_prior_dependent` | 17 |
| `C1_kda_cont` + `C1_kda_disc` | 9 |
| `C1_kda_disc` + `C2` + `C3` | 9 |
| `C2_both_correct_conf` + `C3_prior_dependent` | 8 |
| `C1_kda_cont` + `C1_kda_disc` + `C2` | 3 |
| `C1_kda_cont` alone | 1 |
| `C1_kda_cont` + `C1_kda_disc` + `C3` | 1 |
| all four | 1 |

**OBQA (80 flagged):** `C2_both_correct_conf` alone 67, `C1_kda_disc` + `C2` 9, `C1_kda_disc`
alone 4.

**What happens when an item matches more than one criterion:** nothing special. The criteria form
a **union for pool membership only**. They do not vote on, weight, or constrain the category
label. 95 of 328 SciQ items (29.0%) carry two or more flags and are labelled exactly like
single-flag items.

---

## 4. Stage 2 — category assignment (manual, not reproducible)

Every one of the 408 pooled questions was read by one annotator and given **exactly one** primary
label. Distribution recomputed from the artefact:

| Category | SciQ (n=328) | OBQA (n=80) |
|---|---:|---:|
| `common_knowledge` | 182 | 41 |
| `parametric_knowledge` | 48 | **0** |
| `option_leakage` | 45 | 13 |
| `reasoning_shortcut` | 44 | 16 |
| `material_not_necessary` | 9 | 10 |

### 4.1 Reconstructed operational definitions

Each entry gives the reconstructed criterion, the evidence a reviewer can check, and the verbatim
anchor case from `rq1_test_split_failure_analysis.md`.

---

#### `common_knowledge`

*Criterion (reconstructed):* the gold answer is recoverable from **general education or everyday
reasoning** that a secondary-school student would hold, without the specialised material. The
distinguishing test against `parametric_knowledge` is **who would know it**: if a non-specialist
would, it is common knowledge.

*Anchor — SciQ #650* (`KDA_cont = 0.7420`, rank 12/884):

> **Q:** What disease is the result of unchecked cell division caused by a breakdown of the mechanisms regulating the cell cycle?
> **Options:** diabetes · **cancer** · dementia · gout

Justification quoted verbatim: *"'Uncontrolled cell division = cancer' is the single most widely
known fact in cell biology and appears in every secondary-school curriculum; the distractors
(diabetes, dementia, gout) are not cell-cycle disorders at all."*

---

#### `parametric_knowledge`

*Criterion (reconstructed):* the gold answer is **specialised terminology** that a non-specialist
would *not* know, yet a pretrained model recovers it at high confidence with no passage. The label
asserts the item measures *pretraining recall of a term*, not comprehension.

*Anchor — SciQ #121* (`KDA_cont = 0.7183`):

> **Q:** What are the hormones that cause a plant to grow?
> **Options:** **gibberellins** · pistills · pores · sporozoans

Justification quoted verbatim: *"'Gibberellin' is not general knowledge — it is a specific plant
hormone name from an undergraduate botany syllabus — which is exactly the point: a
110M-parameter sentence encoder that has never seen this passage assigns it 0.649 from
pretraining alone."*

**The `common_knowledge` ↔ `parametric_knowledge` boundary is the acknowledged weak point.** It
turns entirely on the annotator's judgement of what is "school-level", with no operational
threshold. The source document names this as its most fragile distinction.

**OBQA has zero items in this category**, and this is substantive rather than an oversight: OBQA's
`fact1` is a general elementary rule, never a specialised term. The gold answer string appears in
the fact for only 11.4% of OBQA items against 94.8% for SciQ — there is no specialised fact to
have memorised.

---

#### `reasoning_shortcut`

*Criterion (reconstructed):* the item is solvable by a **surface operation on the stem-plus-option
text** — lexical echo, semantic odd-one-out, or an arithmetic/logical pattern the stem itself
supplies — with no domain knowledge engaged. The trigger lives in the **relationship between the
stem and the options**.

*Anchor — SciQ #162* (cue: stem echo):

> **Q:** Digestive enzymes are released, or secreted, by the organs of which body system?
> **Options:** nervous system · endocrine system · urinary system · **digestive system**

Justification: *"The stem contains the literal string 'Digestive'; the gold option is the only one
that repeats it. This is pure lexical matching — no physiology is engaged."*

*Anchor — OBQA #265* (cue: answer-polarity odd-one-out): three distractors are positive-valence
("fine", "happy", "comfortable"), the gold is the only negative outcome.

---

#### `option_leakage`

*Criterion (reconstructed):* the gold is identifiable from a **defect or formatting artifact in
the option list considered on its own**, without reading the stem at all — a duplicated
distractor, a meta-reference to other option letters, a formatting mismatch.

*Anchor — SciQ #587* (`KDA_cont = 0.8419`, the maximum in the split):

> **Options:** carbohydrates · **enzymes** · carbohydrates · proteins

Justification: *"The option list contains 'carbohydrates' **twice**. A duplicated distractor
cannot be the answer … so the item is effectively 1-of-3."*

*Anchor — OBQA #339:* the gold option is the literal string `"B and D"`, a meta-reference to other
option letters.

**Distinguishing `option_leakage` from `reasoning_shortcut`** — the operative test in the anchor
cases is *where the exploitable signal lives*:

| | Signal location | Needs the stem? | Example |
|---|---|---|---|
| `reasoning_shortcut` | Stem ↔ option **relationship** | Yes | stem echo "Digestive" → "digestive system" |
| `option_leakage` | **Within the option list alone** | No | duplicated "carbohydrates"; gold = `"B and D"` |

This test is a reconstruction. It is not stated as a rule anywhere in the source, and SciQ #587 is
explicitly noted in limitation 2 as being *simultaneously* `option_leakage` and
`common_knowledge`, resolved to one label by annotator judgement.

---

#### `material_not_necessary`

*Criterion (reconstructed):* the residual category. The stem is **logically self-sufficient** — it
carries the full inference — and the reference material only restates a step the stem already
forces. Distinct from `common_knowledge` in that no outside knowledge is needed either.

*Anchor — SciQ #291:*

> **Q:** … Two electrons are shared in a single bond; four electrons are shared in a double bond; and six electrons are shared in this?

Justification: *"The stem carries the entire arithmetic progression: 2 → single, 4 → double,
6 → ?. Counting is sufficient; the passage adds only the word 'triple', which the sequence already
forces."*

*Anchor — OBQA #15:* the fact (`as the use of alternative fuels increases, the use of gasoline
will decrease`) is a verbatim restatement of the inference the stem already invites. Primary-model
gain from the fact: `+0.015`.

---

## 5. Structural cues — automated, and separate from the label

`structural_cues` is a **separate automatic field**; it did not determine the category. Its
detector code is not in the repository (§1.2), but the values present in the artefact are:

| Cue | SciQ | OBQA | Meaning (from §4.3 of the source doc) |
|---|---:|---:|---|
| `gold_in_passage` | 317 | 8 | gold answer string occurs in the reference material |
| `longest_gold` | 80 | 35 | gold is the unique longest option |
| `shortest_gold` | 43 | 7 | gold is the unique shortest option |
| `stem_overlap_gold` | 10 | 0 | gold uniquely shares a content word with the stem |
| `dup_distractors` | 4 | 0 | a duplicated option string |
| `numeric_odd` | 1 | 0 | numeric odd-one-out |
| `abs_in_distractor` | 0 | 4 | absolute-quantifier distractor |

The source doc reports these cues are **only weakly enriched** inside the flagged pool on SciQ
(risk ratios 1.13–1.21) and concludes they are "a property of the datasets more than a property of
the questions the models happen to short-circuit". The `head_family3` and `stem_echo` detectors
discussed in §4.3 of that doc do **not** appear as values in the exported artefact.

---

## 6. Mapping the four detection criteria onto the five categories

There is **no mapping**. This is worth stating plainly because the question presupposes one.

- The four criteria (`C1_kda_cont`, `C1_kda_disc`, `C2_both_correct_conf`, `C3_prior_dependent`)
  answer *"is this question suspicious?"* — they are a **union filter for pool membership**.
- The five categories answer *"why is it suspicious?"* — assigned afterwards by reading.

The relationship is empirical, reported as a cross-tabulation in §2.3 of the source doc, and is
many-to-many. Reproduced here from that table (rows sum above the column totals because an item
can carry several flags):

**SciQ**

| Category | `C1_kda_cont` | `C1_kda_disc` | `C2_both_correct_conf` | `C3_prior_dependent` |
|---|---:|---:|---:|---:|
| `common_knowledge` | 9 | 90 | 89 | 59 |
| `parametric_knowledge` | 3 | 30 | 12 | 21 |
| `reasoning_shortcut` | 2 | 23 | 18 | 13 |
| `option_leakage` | 1 | 20 | 19 | 18 |
| `material_not_necessary` | 0 | 4 | 2 | 4 |

No criterion is diagnostic of any category. `C1_kda_disc` and `C2_both_correct_conf` both spread
across all five.

**On multiple matches:** an item matching three criteria receives one label, chosen by reading,
exactly as a one-criterion item does. The flags are preserved in `detection_flags` so a reviewer
can re-stratify.

### 6.1 A correction to the source document

`rq1_test_split_failure_analysis.md` §1 and limitation 4 state that no Setting-C run exists for
OpenBookQA. **That is no longer true.** The following now exist:

```
results/ex2_counterfactual/results_counterfactual_obqa_test_full.json         (500 q, --min-substitution-tier partial)
results/ex2_counterfactual/results_counterfactual_obqa_test_exact_tier.json   (sensitivity run, exact tier only)
docs/ex2_counterfactual/counterfactual_obqa_analysis.md
```

**This has now been fixed.** In the original artefact all 80 OBQA records carried
`counterfactual_class: null` and no OBQA item carried a `C3_prior_dependent` flag. C3 has since
been applied to OpenBookQA with
[`code/ex5_failure_audit/rebuild_flagged_pool.py`](../../code/ex5_failure_audit/rebuild_flagged_pool.py):

| | Before (artefact) | After (C3 applied) |
|---|---:|---:|
| Flagged | 80 (16.0%) | **100 (20.0%)** |
| `C1_kda_disc` | 13 | 13 |
| `C2_both_correct_conf` | 76 | 76 |
| `C3_prior_dependent` | **0** | **26** |
| `counterfactual_class` populated | 0 | 49 of 100 (51 are Setting-C ineligible) |

Of the 80 originally-flagged items, 6 are `prior_dependent`, 18 `context_dependent`, 5
`unstable_other` and 51 ineligible; the other 20 `prior_dependent` items are **newly admitted to
the pool**.

**Coverage is now complete (updated this round).** The 20 new admissions have been labelled:

| Field | Populated | Missing | Note |
|---|---:|---:|---|
| `failure_category` | **100 / 100** | 0 | 80 carried from the original artefact, 20 by a second annotator |
| `counterfactual_class` | 49 / 100 | 51 | **all 51 are pre-existing items, 0 are new admissions** — they are Setting-C *ineligible* (`substitution_tier == "none"`), so this is a structural gap in the counterfactual mechanism, not an annotation debt |

`counterfactual_class` for the 51 cannot be filled by labelling: OBQA's `fact1` is a deductive
rule with no answer span to substitute, so 71% of the whole split is ineligible. See
[`counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) §5.1.

**Category distribution, kept separable by annotator:**

| Category | Original 80 | New 20 | Total 100 |
|---|---:|---:|---:|
| `common_knowledge` | 41 | 15 | 56 |
| `reasoning_shortcut` | 16 | 3 | 19 |
| `option_leakage` | 13 | 0 | 13 |
| `material_not_necessary` | 10 | 2 | 12 |
| `parametric_knowledge` | 0 | 0 | **0** |

> **⚠ Annotator effect.** The 20 new labels were assigned by a **different annotator** (this
> session) than the original 80, applying the *reconstructed* rubric of §4.1 rather than whatever
> the original annotator held in mind. Single pass, no second rater, no κ — the same limitation
> as the original, now compounded by a rater change mid-pool. The `failure_category_status` field
> distinguishes `carried_from_original_artefact` from `second_annotator_new_admission` on every
> record, and per-item rationales are in
> `results/ex5_failure_analysis/obqa_new_admission_labels.json`. **The 56/19/13/12 totals should
> not be quoted as a single distribution** without noting that they mix two annotators.
>
> The one reassuring signal: the new labels reproduce the original's two structural features
> without being tuned to — `common_knowledge` dominant (75% of new vs 51% of original) and
> `parametric_knowledge` still empty, which is independently explained by OBQA having no
> specialised terminology to memorise.

Output: `results/ex5_failure_analysis/rq1_flagged_questions_rebuilt_obqa.json`

```bash
uv run --active python code/ex5_failure_audit/rebuild_flagged_pool.py --dataset obqa --carry-labels \
    --out results/ex5_failure_analysis/rq1_flagged_questions_rebuilt_obqa.json
```

Two caveats before these 100 items are used. First, the 20 new admissions are unlabelled, so the
category mix in §4 still describes the 80-item pool. Second, and more seriously, **70.8% of
OBQA's `context_dependent` labels rest on the loosest `partial` substitution tier**, where the
perturbed passage is frequently ungrammatical — so the OBQA `prior_dependent` count of 26 is best
read as a lower bound. See
[`counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) §9.

Note the OBQA counterfactual run carries its own severe caveat (only 29.0% of the split is
eligible, and the perturbation is often ungrammatical); see
[`docs/ex2_counterfactual/counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) §5.

---

## 7. What a reviewer should conclude

**Trustworthy without qualification** — the population-level findings of the RQ1 study
(§4.1–§4.2 of the source doc: flat zero-context solve rate across `KDA_cont` deciles, positive
`r(KDA, P_wof)`, ~30% of numerator mass from already-correct pairs). These rest on the full
884/500-item populations and on committed, re-runnable code. They do **not** depend on any hand
label.

**Trustworthy as an ordered impression, not as a measurement** — the category mix in §2.1–§2.2.
Single annotator, no rubric written in advance, no second rater, no κ, one forced label per item,
and no script to reproduce the selection. The rank order (`common_knowledge` dominant on both
datasets; `parametric_knowledge` absent on OBQA) is supported by an independent structural fact
(gold-in-fact 11.4% vs 94.8%) and is the part of the taxonomy most likely to survive
re-annotation. The precise percentages are not measurements.

**To make this auditable, in ascending cost:**

1. ~~Commit the selection + structural-cue script so the pool is re-derivable.~~ **Done** —
   [`code/ex5_failure_audit/rebuild_flagged_pool.py`](../../code/ex5_failure_audit/rebuild_flagged_pool.py),
   verified to reproduce the artefact exactly on both datasets. See
   [`provenance_rq1_flagged_questions.md`](provenance_rq1_flagged_questions.md).
2. ~~Re-run criterion C3 on OBQA from the existing counterfactual results.~~ **Done** — see §6.1.
3. Write the rubric down as a decision tree, then have a second annotator label a random ~80-item
   sample and report Cohen's κ, with the `common_knowledge` ↔ `parametric_knowledge` boundary as
   the pre-registered point of concern.
4. Permit multi-label annotation, since limitation 2 records that forcing one primary label
   understates every category except the chosen one.
