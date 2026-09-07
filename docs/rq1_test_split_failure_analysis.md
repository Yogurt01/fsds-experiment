# RQ1 Diagnostic — Systematic Failure Analysis on the SciQ and OpenBookQA Test Splits

**Question under investigation:** *Is the target course material strictly necessary to answer the
multiple-choice question?* — i.e. does `KDA_cont` conflate genuine contextual reliance with
parametric priors, common sense, and superficial answer-selection heuristics?

| | |
|---|---|
| Datasets | `datasets/sciq/sciq_test_full.json` (884 q), `datasets/openbookqa/obqa_test_full.json` (500 q) |
| Metric | `KDA_cont(q) = Σ_m (1 − P_m(R^q=1))·P_m(R^{q+f}=1) / Σ_m (1 − P_m(R^q=1))` |
| Simulated students | `KDA_small` (|M| = 4): `t5-small-ssm-nq`, `kda-albert-xlarge-v2-race`, `kda-mpnet-base-race`, `kda-scibert-uncased-race` |
| Primary model | `Riiid/kda-mpnet-base-race` (strongest; also the primary in `categorize_kda_results.py`) |
| Settings | A = question + options (no fact) · B = + gold fact · C = + counterfactual fact (SciQ only) |
| Source results | `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json`, `results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json`, `results/ex2_counterfactual/results_counterfactual_sciq_test_full.json`, `results/ex1_category_questions/basic_category/` |
| Annotated output | `results/ex5_failure_analysis/rq1_flagged_questions.json` (408 questions, full records) |

> **Scope note (RQ1 only).** This document is a diagnosis. No new metric is proposed, defined,
> or implemented anywhere below.

---

## 0. Executive summary

Three findings, in order of strength.

1. **The failure modes are real and are not rare.** 328 / 884 SciQ questions (37.1%) and
   80 / 500 OpenBookQA questions (16.0%) meet at least one of the Step-1 context-independence
   criteria. Every one of the 408 was read and assigned a primary failure category; all five
   taxonomy categories are populated on SciQ, four of five on OpenBookQA.

2. **The distortion is systematic in the strong sense: `KDA_cont` does not penalise
   context-independence — it very slightly rewards it.** The flagged pool has a *higher* mean
   `KDA_cont` than the rest of the split on both datasets (SciQ 0.491 vs 0.460; OBQA 0.287 vs
   0.268), the `both_correct` bucket scores above the split mean on both (SciQ 0.495 vs 0.471;
   OBQA 0.348 vs 0.271), and zero-context solve rate is essentially **flat across `KDA_cont`
   deciles** (Table 10). The top KDA decile still contains 41.6% (SciQ) / 42.0% (OBQA) of items
   the primary model answers correctly with no passage at all.

3. **The mechanism is identifiable in the metric's own arithmetic.** `KDA_cont` is a
   `(1 − P(R^q=1))`-weighted mean of `P(R^{q+f}=1)`. Because three of the four simulated students
   are near-uniform on Setting A (weight sd 0.041–0.099, Table 11), the weights are nearly
   constant across questions, and the score collapses onto with-fact confidence:
   `r(KDA, P_wf) = +0.86 / +0.88`, while `r(KDA, P_wof) = +0.08 / +0.30` — **positive**, where a
   necessity metric should be strongly negative. 29.8% (SciQ) / 30.8% (OBQA) of the aggregate
   numerator mass is contributed by model–question pairs that were *already correct without the
   fact*.

The honest caveat, stated up front: what is systematic is the metric's *insensitivity* to
context-independence (finding 2–3, which rest on the full 884/500-item populations). The
*taxonomy mix* (finding 1) rests on single-annotator labels with no second rater, and the
structural-cue enrichment inside the flagged pool is weak on SciQ (Table 4). Section 5 lists all
limitations.

---

## 1. Candidate selection (Step 1)

Three operational criteria, applied to the full test splits. A question enters the pool if it
meets **any** of them.

| Flag | Criterion | SciQ | OBQA |
|---|---|---:|---:|
| `C1_kda_cont` | `KDA_cont ≥ 0.70` **and** primary model correct in Setting A | 15 | 0 |
| `C1_kda_disc` | discrete-KDA proxy `= 1.0` **and** primary model correct in Setting A | 167 | 13 |
| `C2_both_correct_conf` | `both_correct` bucket **and** `P(R^q=1) ≥ 0.70` (primary model) | 140 | 76 |
| `C3_prior_dependent` | Setting-C class `prior_dependent` (counterfactual ignored) | 115 | — |
| | **Union (flagged pool)** | **328 (37.1%)** | **80 (16.0%)** |

Three implementation decisions had to be made, and each is a deviation from the literal brief:

- **Discrete KDA.** No human-student annotations exist in this repository, so
  `KDA_disc` is replaced by its model-ensemble analogue
  `KDA_disc_proxy(q) = Σ_m 1[¬correct_wof]·1[correct_wf] / Σ_m 1[¬correct_wof]` (0 if the
  denominator is 0). `= 1.0` means every student that failed without the fact succeeded with it.
- **`P(R^q=1) ≥ 0.70` is read per-model, not on the ensemble average.** On the ensemble average
  the threshold is unreachable: 0 / 884 SciQ questions clear it, because `kda-albert-xlarge-v2-race`
  is near-uniform and drags the mean to ≈0.33. The primary model (`kda-mpnet-base-race`) is used
  instead, where 145 SciQ / 81 OBQA questions clear it.
- **Criterion C3 could not be applied to OpenBookQA.** `results/ex2_counterfactual/results_counterfactual_sciq_test_full.json`
  covers SciQ only (`data_file: sciq_test_full.json`). The OBQA pool is therefore built from two
  criteria instead of three and its 16.0% is **not** directly comparable to SciQ's 37.1%. This is
  the single largest asymmetry in the study.

---

## 2. Quantitative breakdown (Deliverable 1)

Every one of the 408 flagged questions was read and given exactly one primary failure category.
Percentages of split use the full 884 / 500 denominators; the interval is a 95% Wilson interval
on that proportion.

### 2.1 SciQ test split (n = 884)

| Failure category | n | % of flagged pool | % of split | 95% CI |
|---|---:|---:|---:|---|
| `common_knowledge` | 182 | 55.5% | **20.6%** | [18.1, 23.4] |
| `parametric_knowledge` | 48 | 14.6% | **5.4%** | [4.1, 7.1] |
| `option_leakage` | 45 | 13.7% | **5.1%** | [3.8, 6.7] |
| `reasoning_shortcut` | 44 | 13.4% | **5.0%** | [3.7, 6.6] |
| `material_not_necessary` | 9 | 2.7% | **1.0%** | [0.5, 1.9] |
| **Total flagged** | **328** | 100% | **37.1%** | [34.0, 40.3] |

### 2.2 OpenBookQA test split (n = 500)

| Failure category | n | % of flagged pool | % of split | 95% CI |
|---|---:|---:|---:|---|
| `common_knowledge` | 41 | 51.2% | **8.2%** | [6.1, 10.9] |
| `reasoning_shortcut` | 16 | 20.0% | **3.2%** | [2.0, 5.1] |
| `option_leakage` | 13 | 16.2% | **2.6%** | [1.5, 4.4] |
| `material_not_necessary` | 10 | 12.5% | **2.0%** | [1.1, 3.6] |
| `parametric_knowledge` | 0 | 0.0% | **0.0%** | [0.0, 0.8] |
| **Total flagged** | **80** | 100% | **16.0%** | [13.0, 19.5] |

**The empty cell is informative.** OpenBookQA has *zero* `parametric_knowledge` items in the
flagged pool. Its "target fact" is a general elementary rule (`a plant requires sunlight to grow`),
never a specialised term; the gold answer string appears in the fact for only 11.4% of items,
against 94.8% for SciQ. There is simply no specialised fact to have memorised. The two datasets
fail KDA for *different reasons*: SciQ leaks specialised terminology and distractor artifacts,
OpenBookQA leaks common sense and answer-shape.

### 2.3 Category × detection criterion

A question can carry more than one flag, so rows sum above the column totals of §1.

**SciQ**

| Category | `C1_kda_cont` | `C1_kda_disc` | `C2_both_correct_conf` | `C3_prior_dependent` |
|---|---:|---:|---:|---:|
| `common_knowledge` | 9 | 90 | 89 | 59 |
| `parametric_knowledge` | 3 | 30 | 12 | 21 |
| `reasoning_shortcut` | 2 | 23 | 18 | 13 |
| `option_leakage` | 1 | 20 | 19 | 18 |
| `material_not_necessary` | 0 | 4 | 2 | 4 |
| **Total** | **15** | **167** | **140** | **115** |

**OpenBookQA**

| Category | `C1_kda_cont` | `C1_kda_disc` | `C2_both_correct_conf` |
|---|---:|---:|---:|
| `common_knowledge` | 0 | 7 | 40 |
| `reasoning_shortcut` | 0 | 2 | 15 |
| `option_leakage` | 0 | 2 | 12 |
| `material_not_necessary` | 0 | 2 | 9 |
| `parametric_knowledge` | 0 | 0 | 0 |
| **Total** | **0** | **13** | **76** |

The three criteria are not redundant: `C3_prior_dependent` contributes 115 SciQ items that the
accuracy-bucket criteria largely miss, because a question can be *wrong* in Setting A and still
have the model ignore the passage when it is perturbed.

---

## 3. Representative case studies (Deliverable 2)

All metrics are taken verbatim from the results files. `P(R^q=1)` and `P(R^{q+f}=1)` are given
both as the 4-model ensemble mean and for the primary model, because the ensemble mean is
depressed by the near-uniform students.

### 3.1 `common_knowledge`

---

**SciQ #650** — `KDA_cont = 0.7420` (rank 12 of 884) · flags `C1_kda_cont`, `C1_kda_disc`, `C2_both_correct_conf`

> **Q:** What disease is the result of unchecked cell division caused by a breakdown of the mechanisms regulating the cell cycle?
> **Options:** diabetes · **cancer** · dementia · gout
> **Passage:** *"6.3 Cancer and the Cell Cycle — Cancer is the result of unchecked cell division caused by a breakdown of the mechanisms regulating the cell cycle. …"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.7420 | 0.339 / **0.792** | 0.767 / 0.999 |

*Why the passage is not required.* "Uncontrolled cell division = cancer" is the single most widely
known fact in cell biology and appears in every secondary-school curriculum; the distractors
(diabetes, dementia, gout) are not cell-cycle disorders at all. The primary model is already at
0.792 with no passage. Yet the question sits in the top 5% of the split by `KDA_cont` — the metric
reads the jump 0.792 → 0.999 as strong evidence of knowledge dependency, when the residual 0.208
is headroom, not dependency.

---

**OBQA #375** — `KDA_cont = 0.6203` (rank 3 of 500) · flags `C1_kda_disc`, `C2_both_correct_conf`

> **Q:** If you find something smooth and hard on the ground, it is probably made of what?
> **Options:** **minerals** · mist · clouds · water
> **Fact:** *"rock is made of minerals"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.6203 | 0.434 / **0.906** | 0.675 / 0.992 |

*Why the passage is not required.* Mist, clouds and water are all non-solid; "smooth and hard on
the ground" eliminates three of four options by physical state alone, before any mineralogy is
consulted. The fact (`rock is made of minerals`) supplies a link the question does not need.
This is the **third-highest-scoring question in the entire OpenBookQA split**.

### 3.2 `parametric_knowledge`

---

**SciQ #121** — `KDA_cont = 0.7183` (rank 27 of 884) · flags `C1_kda_cont`, `C1_kda_disc`

> **Q:** What are the hormones that cause a plant to grow?
> **Options:** **gibberellins** · pistills · pores · sporozoans
> **Passage:** *"Gibberellins are hormones that cause the plant to grow. …"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.7183 | 0.320 / **0.649** | 0.742 / 0.999 |

*Why the passage is not required.* "Gibberellin" is not general knowledge — it is a specific plant
hormone name from an undergraduate botany syllabus — which is exactly the point: a 110M-parameter
sentence encoder that has never seen this passage assigns it 0.649 from pretraining alone, while
the alternatives (`pistills`, `pores`, `sporozoans`) are not hormones. The item measures whether
the term survived pretraining, not whether the passage was read.

---

**SciQ #880** — `KDA_cont = 0.3019` · flag `C2_both_correct_conf`

> **Q:** What do you call the ancient cores of continents, where the earliest continental crust is now found?
> **Options:** **cratons** · craters · escarpments · mantles
> **Passage:** *"The earliest continental crust is now found in the ancient cores of continents, called the cratons."*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.3019 | 0.339 / **0.845** | 0.433 / 1.000 |

*Why the passage is not required.* `craton` is specialist geology vocabulary, and the primary model
recovers it at 0.845 with no context. This case also shows the *reverse* failure: because the two
near-uniform students never resolve it, the ensemble `KDA_cont` lands at 0.30 — below the split
mean — so the metric ranks a memorised, passage-free item as *low* knowledge-dependency purely
because the weak students were weak. The score is tracking student capability, not question
property.

### 3.3 `reasoning_shortcut`

---

**SciQ #162** — `KDA_cont = 0.2938` · flag `C2_both_correct_conf` · cue: stem echo

> **Q:** Digestive enzymes are released, or secreted, by the organs of which body system?
> **Options:** nervous system · endocrine system · urinary system · **digestive system**
> **Passage:** *"Digestive enzymes are released, or secreted, by the organs of the digestive system. …"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.2938 | 0.375 / **0.815** | 0.466 / 0.992 |

*Why the passage is not required.* The stem contains the literal string "Digestive"; the gold
option is the only one that repeats it. This is pure lexical matching — no physiology is engaged.
The question is a cloze deletion of the passage's first sentence, which is how 34.0% of SciQ test
questions are built (Table 12), and the deleted span is the one word the stem already supplies.

---

**OBQA #265** — `KDA_cont = 0.1822` · flag `C2_both_correct_conf` · cue: answer-polarity odd-one-out

> **Q:** If your dog sits in an oxygen deficient chamber, what happens?
> **Options:** it will be fine · it will be happy · it will be comfortable · **It will pass out**
> **Fact:** *"an animal requires oxygen for to breathe"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.1822 | 0.415 / **0.975** | 0.396 / 0.985 |

*Why the passage is not required.* Three distractors are positive-valence and mutually
interchangeable ("fine", "happy", "comfortable"); the gold is the only negative outcome and the
only one with different capitalisation. A solver that has never heard of oxygen picks it. Note
`P(R^{q+f}=1) − P(R^q=1) = +0.010` on the primary model: the fact adds essentially nothing, and
the metric correctly returns a low 0.182 — but only because *both* settings are saturated, not
because the metric detected the shortcut.

### 3.4 `option_leakage`

---

**SciQ #587** — `KDA_cont = 0.8419` — **the maximum `KDA_cont` in the entire SciQ test split** · flag `C3_prior_dependent`

> **Q:** What are catalysts in living things called?
> **Options:** carbohydrates · **enzymes** · carbohydrates · proteins
> **Passage:** *"…Catalysts in living things are called enzymes. …"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary | Setting C |
|---:|---:|---:|---|
| 0.8419 | 0.492 / 0.367 | 0.884 / 0.999 | `prior_dependent`, `kda_adjusted_hard = 0.839` |

*Why the passage is not required.* The option list contains **"carbohydrates" twice**. A duplicated
distractor cannot be the answer (an MCQ with two identical options has no unique gold at that
index), so the item is effectively 1-of-3 with one implausible remaining distractor. Beyond the
artifact, "catalysts in living things = enzymes" is standard school biology. The counterfactual run
confirms the diagnosis independently: when the passage is rewritten so it asserts a *different*
answer, the ensemble **ignores it and still answers "enzymes"** (`prior_dependent`). The
highest-scoring question in the dataset — the item `KDA_cont` nominates as its most
knowledge-dependent — is a broken item solved from priors.

---

**OBQA #339** — `KDA_cont = 0.3165` (above the 0.271 split mean) · flag `C2_both_correct_conf`

> **Q:** How do polar bears survive the cold?
> **Options:** **B and D** · Double Fur Coats · Cold blooded · Compact ears
> **Fact:** *"polar bears live in cold environments"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.3165 | **0.765** / 0.990 | 0.538 / 0.999 |

*Why the passage is not required.* The gold option is the string `"B and D"` — a meta-reference to
other option letters, structurally unlike every distractor. This is the **highest zero-context
ensemble confidence in the whole OpenBookQA split (0.765)**, and it is entirely an artifact of
option formatting. Note also `P(R^{q+f}=1) = 0.538 < P(R^q=1) = 0.765` on the ensemble: adding the
fact made the ensemble *worse*, and the question still scores above the split mean.

### 3.5 `material_not_necessary`

---

**SciQ #291** — `KDA_cont = 0.2694` · flag `C3_prior_dependent`

> **Q:** Each bond includes a sharing of electrons between atoms. Two electrons are shared in a single bond; four electrons are shared in a double bond; and six electrons are shared in this?
> **Options:** magnetic bond · **triple bond** · quadruple bond · ionic bond
> **Passage:** *"Each bond includes a sharing of electrons between atoms. Two electrons are shared in a single bond; four electrons are shared in a double bond; and six electrons are shared in a triple bond."*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary | Setting C |
|---:|---:|---:|---|
| 0.2694 | 0.450 / 0.348 | 0.439 / 0.270 | `prior_dependent`, `kda_adjusted_hard = 0.080` |

*Why the passage is not required.* The stem carries the entire arithmetic progression: 2 → single,
4 → double, 6 → ?. Counting is sufficient; the passage adds only the word "triple", which the
sequence already forces. The counterfactual run agrees: the perturbed passage is ignored
(`prior_dependent`), and adjusting for that collapses the score from 0.269 to 0.080.

---

**OBQA #15** — `KDA_cont = 0.2404` · flags `C1_kda_disc`, `C2_both_correct_conf`

> **Q:** As gasoline costs rise, alternative fuels are being used, which means that
> **Options:** wind power will be expensive · gas costs will rise · oil costs will be maintained · **gasoline will be needed less**
> **Fact:** *"as the use of alternative fuels increases, the use of gasoline will decrease"*

| `KDA_cont` | `P(R^q=1)` ens / primary | `P(R^{q+f}=1)` ens / primary |
|---:|---:|---:|
| 0.2404 | 0.582 / **0.929** | 0.482 / 0.944 |

*Why the passage is not required.* The "fact" is a verbatim restatement of the inference the stem
already invites — substituting one fuel for another means using less of the first. The stem also
rules out `gas costs will rise` (it is a premise, not a consequence). Primary-model gain from the
fact: `+0.015`.

---

## 4. Systematicity assessment (Deliverable 3)

### 4.1 The metric's response to context-independence

This is the core test, and it runs on the **full 884 / 500-item populations**, not on the annotated
subset — so it is not exposed to annotation error.

**Table 3 — mean `KDA_cont`, flagged vs unflagged**

| | flagged mean | unflagged mean | direction |
|---|---:|---:|---|
| SciQ | **0.491** (n=328) | 0.460 (n=556) | flagged score *higher* |
| OBQA | **0.287** (n=80) | 0.268 (n=420) | flagged score *higher* |

**Table 5 — mean `KDA_cont` by baseline bucket (pooled-probability ensemble vote)**

| Dataset | `both_correct` | `wrong_to_correct` | `both_wrong` | `correct_to_wrong` | `both_correct` mean KDA | split mean KDA |
|---|---:|---:|---:|---:|---:|---:|
| SciQ | 390 (44.1%) | 393 (44.5%) | 89 (10.1%) | 12 (1.4%) | **0.495** | 0.471 |
| OBQA | 117 (23.4%) | 86 (17.2%) | 240 (48.0%) | 57 (11.4%) | **0.348** | 0.271 |

44.1% of the SciQ test split is answered correctly **without the passage** by the pooled ensemble,
and that bucket receives an *above-average* `KDA_cont`. On OpenBookQA the effect is larger: the
`both_correct` bucket scores 0.348 against a 0.271 split mean, a +28% relative premium for
questions that demonstrably did not need the fact.

**Table 10 — zero-context solve rate by `KDA_cont` decile (primary model, Setting A)**

| Decile | SciQ KDA range | SciQ acc. w/o fact | OBQA KDA range | OBQA acc. w/o fact |
|---|---|---:|---|---:|
| D1 (lowest) | 0.085–0.269 | 51.1% | 0.077–0.151 | 16.0% |
| D2 | 0.269–0.342 | 69.3% | 0.152–0.173 | 36.0% |
| D3 | 0.343–0.392 | 57.3% | 0.174–0.198 | 52.0% |
| D4 | 0.394–0.438 | 56.8% | 0.198–0.221 | 46.0% |
| D5 | 0.438–0.480 | 42.7% | 0.221–0.247 | 48.0% |
| D6 | 0.480–0.514 | 50.0% | 0.247–0.276 | 44.0% |
| D7 | 0.514–0.557 | 46.6% | 0.277–0.303 | 48.0% |
| D8 | 0.557–0.602 | 48.3% | 0.304–0.360 | 46.0% |
| D9 | 0.603–0.656 | 50.0% | 0.360–0.433 | 56.0% |
| D10 (highest) | 0.656–0.842 | **41.6%** | 0.437–0.672 | **42.0%** |

If `KDA_cont` measured necessity of the material, this column would fall steeply from D1 to D10.
It does not. It is flat and non-monotone, and the highest-scoring decile — the questions the metric
nominates as *most* knowledge-dependent — is still solved without any passage more than 40% of the
time on both datasets.

**Table 9 — the high-KDA tail is *enriched* for flagged items**

| Dataset | tail | flagged in tail | base rate | category mix of the flagged tail |
|---|---|---:|---:|---|
| SciQ | top decile (n=88) | 45.5% | 37.1% | 23 common · 8 parametric · 6 option-leakage · 3 shortcut |
| SciQ | `KDA_cont ≥ 0.70` (n=35) | **51.4%** | 37.1% | 11 common · 3 parametric · 2 option-leakage · 2 shortcut |
| OBQA | top decile (n=50) | 22.0% | 16.0% | 8 common · 2 option-leakage · 1 shortcut |

Over half of the SciQ questions the metric scores at `KDA_cont ≥ 0.70` are in the flagged pool.

### 4.2 The mechanism

**Table 7 — what `KDA_cont` actually correlates with**

| | `r(KDA, P_wof)` | `r(KDA, P_wf)` | `r(KDA, gain)` |
|---|---:|---:|---:|
| SciQ | **+0.075** | +0.863 | +0.802 |
| OBQA | **+0.301** | +0.878 | +0.684 |

(Spearman ρ agrees: +0.084 / +0.847 and +0.370 / +0.878.)

**Table 11 — weight dispersion per simulated student, `weight = 1 − P(R^q=1)`**

| Model (SciQ) | weight mean | weight **sd** | `P(R^{q+f}=1)` mean | sd |
|---|---:|---:|---:|---:|
| `t5-small-ssm-nq` | 0.616 | 0.397 | 0.606 | 0.399 |
| `kda-albert-xlarge-v2-race` | 0.749 | **0.044** | 0.274 | 0.092 |
| `kda-mpnet-base-race` | 0.591 | 0.256 | 0.815 | 0.262 |
| `kda-scibert-uncased-race` | 0.718 | **0.091** | 0.425 | 0.183 |

Two of four students (three of four on OBQA) contribute a weight that is nearly a **constant**
across all 884 questions. The denominator `Σ_m (1 − P_m(R^q=1))` therefore barely varies, and
`KDA_cont` degenerates into an almost-unweighted mean of `P_m(R^{q+f}=1)` — a measure of how easy
the question is *once you have the fact in front of you*. That is precisely the quantity that is
high for common-knowledge items, memorised terminology, stem-echo items and leaky option sets.

**Table 8 — numerator mass from already-solved pairs**

| Dataset | share of aggregate numerator from `is_correct_without_fact = True` pairs | questions where that share > 50% |
|---|---:|---:|
| SciQ | **29.8%** | 256 / 884 (29.0%) |
| OBQA | **30.8%** | 133 / 500 (26.6%) |

Roughly three tenths of the evidence `KDA_cont` accumulates comes from student–question pairs that
did not need the fact.

**Table 6 — counterfactual corroboration (SciQ, Setting C)**

| Class | n | % of 860 eligible |
|---|---:|---:|
| `context_dependent` | 669 | 77.8% |
| `prior_dependent` | 115 | **13.4%** |
| `unstable_other` | 76 | 8.8% |

On the 115 `prior_dependent` items the ensemble keeps the gold answer even when the passage is
rewritten to assert a different one. Their mean `kda_original` is **0.474** — statistically
indistinguishable from the split mean of 0.477 — while `kda_adjusted_hard` drops to **0.208**.
The unadjusted metric assigns average knowledge-dependency to questions that provably ignore the
knowledge. Their category mix is 59 `common_knowledge`, 21 `parametric_knowledge`,
18 `option_leakage`, 13 `reasoning_shortcut`, 4 `material_not_necessary`.

### 4.3 Where the evidence is weaker

**Structural cues are only mildly enriched inside the flagged pool (Table 4).**

| Cue | SciQ flagged | SciQ split | RR | OBQA flagged | OBQA split | RR |
|---|---:|---:|---:|---:|---:|---:|
| gold is unique longest option | 24.4% | 20.8% | 1.17 | 43.8% | 32.0% | **1.37** |
| gold is unique shortest option | 13.1% | 14.6% | 0.90 | 8.8% | 10.2% | 0.86 |
| stem-echo (gold uniquely shares a content word with the stem) | 2.7% | 2.3% | 1.21 | 0.0% | 1.0% | 0.00 |
| templated option family (≥3 options share a head word) | 22.6% | 20.0% | 1.13 | 1.2% | 4.6% | 0.27 |
| duplicated option string | 1.2% | 0.6% | 2.16 | 0.0% | 0.0% | — |

Risk ratios of 1.13–1.21 on SciQ are not a strong enrichment signal. The honest reading is that
`reasoning_shortcut` and `option_leakage` on SciQ are **prevalent at the dataset level rather than
concentrated in the flagged pool** — 20.0% of the whole SciQ test split has a templated option
family, whether or not the models exploit it on that particular item.

**Table 13 — longest-option bias against chance, whole split**

| Dataset | gold is unique longest | chance | z |
|---|---:|---:|---:|
| SciQ | 184 / 695 = 26.5% | 25% | 0.90 (n.s.) |
| OBQA | 160 / 427 = **37.5%** | 25% | **5.95** |

The longest-option heuristic is a genuine, highly significant artifact of **OpenBookQA** and is
**not** present in SciQ. Claims about "longest option bias" must be made per-dataset.

**Table 12 — item construction**

| Diagnostic | value |
|---|---:|
| SciQ: question content-words ≥80% contained in the passage's **first sentence** | 301 / 884 (34.0%) |
| SciQ: ≥80% contained in **some single** passage sentence | 494 / 884 (55.9%) |
| SciQ: gold answer string present in passage | 838 / 884 (94.8%) |
| OBQA: question content-words ≥80% contained in `fact1` | 62 / 500 (12.4%) |
| OBQA: gold answer string present in `fact1` | 57 / 500 (11.4%) |

More than half of SciQ test questions are near-verbatim cloze deletions of one passage sentence.
This is the structural root of both the `reasoning_shortcut` (stem echo) and the inflated
Setting-B probabilities that dominate `KDA_cont`: with the passage in the prompt, Setting B is
often a string-matching task rather than a comprehension task.

### 4.4 Verdict

**Yes — parametric and heuristic shortcuts systematically distort `KDA_cont` on both test
benchmarks, but the distortion is one of *insensitivity*, not of *inversion*.**

Substantiated as systematic:

- `KDA_cont` carries **no usable signal about whether the material is necessary**. Zero-context
  solve rate is flat across its deciles (Table 10); `r(KDA, P(R^q=1))` is positive, not negative
  (Table 7); the `both_correct` bucket and the flagged pool both score *above* the split mean
  (Tables 3, 5). This holds on the full populations of both datasets and does not depend on any
  hand annotation.
- The cause is structural and reproducible from the formula: near-constant weights from
  under-powered simulated students collapse `KDA_cont` onto `P(R^{q+f}=1)` (Table 11), and ~30% of
  the numerator mass comes from pairs already solved without the fact (Table 8).
- Context-independent questions are **abundant, not exceptional**: 37.1% of SciQ and 16.0% of OBQA
  meet an explicit context-independence criterion, and 44.1% / 23.4% are solved outright without
  the passage. The high-KDA tail is enriched for them, not depleted (Table 9).
- Independent confirmation on SciQ: 13.4% of eligible items are `prior_dependent` under
  counterfactual perturbation, and they receive an entirely average `KDA_cont` (0.474 vs 0.477).

Substantiated as systematic but **dataset-specific**:

- **SciQ** fails through *construction*: 55.9% cloze-from-passage items, 20.0% templated option
  families, and specialised terminology recoverable from pretraining. Its dominant modes are
  `common_knowledge` (20.6% of split) and a roughly equal split of `parametric_knowledge` /
  `option_leakage` / `reasoning_shortcut` (≈5% each).
- **OpenBookQA** fails through *answer shape and triviality*: a significant longest-option bias
  (37.5% vs 25% chance, z = 5.95), absurd or joke distractors, and "facts" that restate the
  inference the stem already forces. It has **zero** `parametric_knowledge` failures — there is no
  specialised fact to memorise.

**Reported transparently as *not* strongly supported:**

- The claim that surface heuristics are *concentrated* in the model-flagged pool. Risk ratios on
  SciQ are 1.13–1.21 — weak. The cues are a property of the datasets more than a property of the
  questions the models happen to short-circuit.
- Any cross-dataset comparison of the 37.1% vs 16.0% headline rates. The SciQ pool had three
  detection criteria available and OpenBookQA only two (no counterfactual run exists for OBQA).
- `material_not_necessary` on SciQ (9 items, 1.0% of split, CI [0.5, 1.9]) is a small enough cell
  that its rate should be treated as an order-of-magnitude estimate only.

---

## 5. Limitations

1. **Single annotator, no inter-annotator agreement.** All 408 category labels were assigned by one
   annotator in one pass. No second rater, no adjudication protocol, no κ. The four-way split among
   `common_knowledge` / `parametric_knowledge` / `reasoning_shortcut` / `option_leakage` is the most
   fragile part of the study; the `common_knowledge` ↔ `parametric_knowledge` boundary in particular
   is a judgement about what counts as "school-level", and a different annotator would move items
   across it. The population-level results in §4.1–4.2 do **not** depend on these labels.
2. **Single primary label per question.** Many items exhibit two or three failure modes at once
   (SciQ #587 is simultaneously `option_leakage` and `common_knowledge`). Forcing one primary label
   understates every category except the one chosen. `structural_cues` in the exported JSON
   preserves the automatic secondary signals.
3. **"Answerable without the material" is operationalised through weak models.** The simulated
   students are 30M–110M parameters. Their Setting-A success is a **lower bound** on what a human
   student or a frontier model would solve without context, so the flagged pool of 328/80 almost
   certainly *undercounts*. No human-subject data and no strong-LLM zero-context evaluation were
   run here.
4. **No counterfactual (Setting C) run exists for OpenBookQA.** Criterion C3 is SciQ-only. The two
   pool sizes are therefore not comparable, and the OBQA analysis lacks the strongest single piece
   of evidence available on SciQ.
5. **Discrete KDA is a model proxy, not human `KDA_disc`.** The reference metric is defined over
   student responses; no student annotations exist in this repository.
6. **Thresholds are conventions.** `KDA_cont ≥ 0.70` and `P(R^q=1) ≥ 0.70` come from the task
   specification, not from a calibration. `KDA_cont ≥ 0.70` selects only 35 SciQ and 0 OBQA items,
   because the two splits have very different score scales (mean 0.471 vs 0.271) — a fixed absolute
   threshold is not comparable across datasets. Decile-based analyses (Tables 9, 10) are given
   alongside precisely because they are threshold-free.
7. **`P(R^q=1) ≥ 0.70` was applied to the primary model, not the ensemble.** On the ensemble average
   the criterion is vacuous (0/884). This is a documented deviation, and it makes the C2 pool
   `kda-mpnet-base-race`-specific.
8. **The structural detectors are crude.** `head_family3` (≥3 options sharing a final content word)
   fires on legitimate items as well as templated ones; `stem_echo` requires an exact content-word
   match and misses morphological cues (`osteoporosis` → `bone`, `hepatitis` → `liver`,
   `hyperthermophile` → `heat`), which were caught by reading but are not in the automatic counts.
9. **Correlational, not causal.** Nothing here isolates *why* an individual model answered as it
   did; attention/attribution analysis was not performed.

---

## 6. Reproducing this analysis

Inputs (all read-only):

```
datasets/sciq/sciq_test_full.json
datasets/openbookqa/obqa_test_full.json
results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json
results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json
results/ex2_counterfactual/results_counterfactual_sciq_test_full.json
results/ex1_category_questions/basic_category/categorized_summary.json
```

Output:

```
results/ex5_failure_analysis/rq1_flagged_questions.json
```

The export holds all 408 flagged questions with `question_id`, `question`, `options`,
`gold_answer`, `reference_passage`, `kda_cont`, `p_rq_without_fact_{ensemble,primary}`,
`p_rqf_with_fact_{ensemble,primary}`, `pooled_probs_without_fact`, `detection_flags`,
`failure_category`, `structural_cues`, `counterfactual_class` and `kda_adjusted_hard`. A `schema`
block at the top of the file documents every field.
