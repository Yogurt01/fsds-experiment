# Benchmark-Wide Corpus-Defect Audit — Question/Passage Overlap

**New this round.** Nothing in this document was reported previously; it supersedes the *scoping
note* that proposed this screen in
[`unexploited_buckets_analysis.md`](unexploited_buckets_analysis.md) §6.

| | |
|---|---|
| Scope | **All 1,384 test items** — SciQ 884 + OpenBookQA 500 — independent of any model's performance |
| Script | [`code/ex5_failure_audit/passage_overlap_audit.py`](../../code/ex5_failure_audit/passage_overlap_audit.py) |
| Hand-check | `results/ex5_failure_audit/passage_overlap_handcheck.json` |
| Distribution data | `results/ex5_failure_audit/passage_overlap_audit.json` |

```bash
uv run --active python code/ex5_failure_audit/passage_overlap_audit.py
```

---

## 0. Headline

> ⚠ **SUPERSEDED for SciQ — see [§8](#8-adjudicated-result--the-double-annotated-census-2026-09-09).**
> The figures below are the original **single-annotator** round. After a second, human annotator
> censused the union screen blind and item #584 was adjudicated `ON_TOPIC`, the confirmed count is
> **12 / 884 = 1.36%** (n = 15, κ = 0.857, agreement 0.933). OpenBookQA is unchanged.

| Dataset | Confirmed topic mismatches | Rate | Basis |
|---|---:|---:|---|
| **SciQ** | ~~13 / 884~~ → **12 / 884** (§8) | ~~1.47%~~ → **1.36%** (§8) | census of all 21 items below 0.40 overlap + 20-item sample above; re-censused and double-annotated in §8 |
| **OpenBookQA** | **0 / 500** | **0%** (95% CI upper ≈ 3.2% of split) | 25-item sample of the 123 zero-overlap items |

Three results that change how the earlier findings should be read:

1. **SciQ has a real, small, fully enumerable corpus defect**: 13 items whose support passage is
   about an unrelated subject — **12 after adjudication (§8)**. All are detected by the screen.
2. **OpenBookQA has no such defect at all**, and its zero-overlap mass — 24.6% of the split — is
   entirely by design. The detector does not transfer between the two benchmarks, for the same
   structural reason the counterfactual mechanism did not.
3. **The earlier SciQ H3 finding (5 of 7) is not in line with the base rate — it is enriched
   48.6×** over it (**52.6× on the adjudicated base rate, §8.4**). See §5.

---

## 1. Metric

```
overlap(q) = |content_words(question) ∩ content_words(passage)| / |content_words(question)|
```

Content words are lower-cased alphanumeric tokens of length > 1 with stopwords removed
(`content_tokens`, shared with the pool rebuild), plus **singular/plural stemming**.

**Stemming is not cosmetic.** Without it the detector produces false positives on pure number
differences. Three of the fifteen SciQ items below 0.10 unstemmed have passages that are exactly
on topic:

| Item | Question phrasing | Passage phrasing | Unstemmed | Stemmed |
|---|---|---|---:|---:|
| #201 | "A collapsing **nebula**" | "**Nebulas** collapse until nuclear fusion starts" | 0.000 | 0.250 |
| #251 | "an **invertebrate**, like a **snail**" | "**Snails** are an example of **invertebrates**" | 0.000 | 0.250 |
| #546 | "What is an **adaptation**?" | "**Adaptations** are favorable traits" | 0.000 | **1.000** |

Every genuine topic mismatch stays at 0.000 under either variant, so stemming raises precision at
no cost to recall. All figures below use the stemmed metric.

---

## 2. Distributions — a tail on SciQ, a continuum on OBQA

### 2.1 SciQ (n = 884), passage field = `support`

| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p99 | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.500 | 0.600 | 0.750 | 0.900 | 1.000 | 1.000 | 1.000 | **0.850** |

```
[0.0,0.1)    12 (  1.4%) #
[0.1,0.2)     0 (  0.0%)
[0.2,0.3)     4 (  0.5%)
[0.3,0.4)     5 (  0.6%)
[0.4,0.5)    13 (  1.5%) #
[0.5,0.6)    48 (  5.4%) ###
[0.6,0.7)    74 (  8.4%) #####
[0.7,0.8)    94 ( 10.6%) ######
[0.8,0.9)   187 ( 21.2%) #############
[0.9,1.0]   447 ( 50.6%) ##############################
```

**This is a genuine isolated tail, not a continuum.** 50.6% of the split sits in the top decile,
the region between 0.10 and 0.40 holds just **9 items**, and there is a distinct spike of **11
items at exactly 0.000**. That shape is what makes a threshold meaningful here — and it is what
OpenBookQA lacks.

### 2.2 OpenBookQA (n = 500), passage field = `fact1`

| p1 | p5 | p10 | p25 | p50 | p75 | p90 | p99 | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.000 | 0.000 | 0.053 | 0.333 | 0.667 | 1.000 | 1.000 | **0.374** |

```
[0.0,0.1)   135 ( 27.0%) ################
[0.1,0.2)    37 (  7.4%) ####
[0.2,0.3)    64 ( 12.8%) ########
[0.3,0.4)    52 ( 10.4%) ######
[0.4,0.5)    14 (  2.8%) ##
[0.5,0.6)    65 ( 13.0%) ########
[0.6,0.7)    49 (  9.8%) ######
[0.7,0.8)    18 (  3.6%) ##
[0.8,0.9)    10 (  2.0%) #
[0.9,1.0]    56 ( 11.2%) #######
```

**No tail — zero overlap is the modal value**, at 123 items (24.6%). A single threshold cannot
serve both splits.

### 2.3 Why the metric does not mean the same thing on the two benchmarks

The same structural difference that broke the counterfactual mechanism
([`counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) §5.1)
governs this audit:

| | SciQ `support` | OBQA `fact1` |
|---|---|---|
| Nature | **Extractive** textbook paragraph containing the answer sentence | **Deductive** one-clause general rule |
| Relation to the question | Near-verbatim; 55.9% of items are cloze deletions of one passage sentence | The question asks for an *instance* of the rule |
| Expected overlap | High — mean 0.850 | Low — mean 0.374, modal 0.000 |
| Zero overlap means | **Anomalous → likely defect** | **Normal → by design** |

---

## 3. Hand-check — SciQ

Census of all 21 items below 0.40, plus a seeded 20-item sample above it.

> §3 records the **single-annotator** hand-check as performed. §8 supersedes its SciQ counts.

| Category | n | Share of the 21 examined |
|---|---:|---:|
| `topic_mismatch` | **13** → **12** after adjudication (§8.3) | 61.9% → 57.1% |
| `legitimate_low_overlap` | 8 | 38.1% |
| `orthographic_defect` | 0 | — |
| `other` | 0 | — |

### 3.1 `topic_mismatch` — 13 items

The passage is about an unrelated subject and cannot answer the question:

| Item | Question is about | Passage is about |
|---|---|---|
| #23 | osteoporosis / bone fractures | fungi and the plant kingdom |
| #153 | hydrometers, specific gravity | Mendel's pea-plant crosses |
| #203 | gas pressure in a rigid container | car acceleration (one sentence) |
| #306 | what reptile skin is covered in | the cell cycle and mitosis |
| #412 | isotopes | stoneworts and the first plants |
| #598 | simple diffusion through lipids | blood disorders |
| #610 | the definition of pollution | bread mould, plus image credits |
| #655 | what moving charged particles generate | paint toxicity and disposal |
| #739 | sickle-cell disease | metalloids |
| #746 | what kPa/atm/mmHg measure | lymph organs, plus image credits |
| #754 | what separates river basins | compressive stress on rock |
| ~~**#584**~~ | the heart's atrium | bronchi and respiratory bronchioles — **overturned: adjudicated `ON_TOPIC` (§8.3)**; the passage also covers the heart's atria and ventricles |
| **#602** | how the water cycle ends | domestic hot-water and warm-air heating |

**#584 and #602 are the important ones**: both sit *above* the 0.10 threshold (0.286 and 0.333)
and would have been missed by it. (**#584 was subsequently overturned — see §8.3.** #602 stands,
and on its own still justifies the 0.40 operating point.) They score non-zero only on incidental words — "respiratory"
and "parts" for #584, "water" and "heat" for #602. This is why the operating point is 0.40, not
0.10.

Several passages are not merely off-topic but are not textbook prose at all — #610 and #746 are
substantially **image credit lines** ("Left: Ciar; Right: Jeff Keacher", "Bone: CC-BY-NC-SA 3.0"),
which suggests the defect originates in the corpus-construction scrape.

### 3.2 `legitimate_low_overlap` — 8 items

- **#127 (0.091)** — the only genuine paraphrase case. Gold `bar graph`; the passage really is
  about bar graphs, but the question is a definitional rewording ("a diagram in which the
  numerical values of variables are represented by the height or length of lines or rectangles of
  equal width") that happens to share almost no vocabulary with it.
- **#201, #251, #546** — the stemming artifacts of §1, on topic throughout.
- **#321, #525, #566, #619, #828** — all on topic (farming/erosion, the reaction arrow symbol,
  acids and bases, aerobic vs anaerobic respiration, type 1 diabetes), simply phrased with
  different vocabulary.

### 3.3 Detector performance

| Threshold | Precision | Recall (within examined region) | Pool size |
|---|---:|---:|---:|
| < 0.10 | 0.917 | 0.846 | 12 |
| **< 0.40** | 0.619 | **1.000** | **21** |

**Recommended operating point: < 0.40 with a full census of the ~21 items.** Precision does not
matter at this scale — reading 21 items to find 13 defects costs minutes.

Above 0.40: 0 of 20 sampled items were mismatched, and a passage sharing ≥40% of a question's
content words is structurally unlikely to be off-topic. Both known misses sit below that line.

---

## 4. Hand-check — OpenBookQA

Seeded random sample of 25 of the 123 items at exactly 0.000 overlap.

| Category | n |
|---|---:|
| `topic_mismatch` | **0** |
| `legitimate_low_overlap` | **25** |

**Every sampled pair is a valid rule/instance link.** Representative:

| Item | `fact1` | Question | Verdict |
|---|---|---|---|
| #245 | `a thermal insulator slows the transfer of heat` | why a duck-feather jacket works in snow | valid |
| #430 | `adding salt to a solid decreases the freezing point of that solid` | how to reduce ice on a sidewalk | valid |
| #480 | `a complete revolution of the Earth around the sun takes one solar year` | what occurs once between Jan 1 and Dec 31 | valid |
| #212 | `cows only eat plants` | what barnyard bovines eat | valid |

None shares a content word with its question; all are correct.

**Statistics.** 0/25 gives a 95% Wilson interval of **[0.0%, 13.3%]** on the defect rate *within
the zero-overlap stratum* — an upper bound of ~16 items in the split, with a point estimate of 0.
The interval is wide because a zero numerator is hard to bound tightly; the substantive claim is
supported less by the interval than by the structural argument in §2.3, which predicts the
observed rate.

One minor note rather than a defect: **#255**'s fact reads `alloys are made of two or more metals`
for a gold answer of `iron and carbon` — carbon is not a metal. On topic, so counted as
legitimate, but the fact is imprecise.

---

## 5. Does this change the earlier SciQ H3 finding?

**Yes — it makes it much stronger. 5 of 7 is dramatically enriched, not in line with the base rate.**

| | Value | Adjudicated (§8) |
|---|---:|---:|
| Corpus base rate of topic mismatch (SciQ) | 1.47% | **1.36%** |
| Rate inside the H3 screen pool (`both_wrong` ∧ Setting-C-ineligible, n=7) | **71.4%** | unchanged |
| **Enrichment** | 48.6× | **52.6×** |

The two screens are closely related, which explains the size of the effect and should be stated
plainly rather than treated as an independent confirmation — a passage about the wrong topic will
not contain the gold answer, so it is automatically Setting-C-ineligible:

| Relationship | Value | Adjudicated (§8) |
|---|---|---|
| Mismatches that are Setting-C ineligible | 12 of 13 (92.3%) | **12 of 12 (100%)** |
| Setting-C ineligible items that are mismatches | 12 of 24 (50.0%) | unchanged |
| Items with overlap < 0.40 that are mismatches | 13 of 21 (61.9%) | **12 of 21 (57.1%)** |
| **Union of the two screens** | **33 unique items, capturing 13 of 13 mismatches** | **capturing 12 of 12** |

So Setting-C ineligibility is a 92%-sensitive proxy for topic mismatch on SciQ, and the overlap
detector is the more precise of the two. Their union is a complete detector at this scale.

---

## 6. Revised recommendation for the `both_wrong` re-annotation

This audit was proposed as a prerequisite to the 120-item, 6–7 hour re-annotation scoped in
[`unexploited_buckets_analysis.md`](unexploited_buckets_analysis.md) §6. **It substantially
reduces that plan.**

**What no longer needs a sample at all.** The corpus-defect question is answered by enumeration,
not estimation. On SciQ the 13 mismatches are *all* of them within the examined region, found by
a screen that costs seconds; on OBQA there are none to find. No sampling is required for either.

**What the mis-key question is now worth.** Across 55 items read in the previous round (7 SciQ H3
+ 48 OBQA complement) there were **0 unambiguous mis-keys and 1 contested key** (OBQA #228,
`insects` vs `invertebrates`). At a true rate near 2%, estimating it to ±2 pp needs n ≈ 190 and to
±5 pp needs n ≈ 30 — but a ±5 pp interval around 2% is compatible with 0%, so the exercise cannot
distinguish "rare" from "absent". **Recommendation: drop the mis-key rate estimation.** It was the
original motivation for the 120-item sample and it is not answerable at affordable n.

**Revised plan — 33 items, ~1.5 hours, replacing 120 items and 6–7 hours:**

| Step | Scope | Effort |
|---|---|---|
| 1. Census the SciQ union screen (overlap < 0.40 ∪ Setting-C-ineligible) | 33 items | ~45 min |
| 2. Double-annotate 15 of the 33, report Cohen's κ | 15 items | ~20 min |
| 3. Adjudicate and publish a **corpus mismatch rate** for SciQ | — | ~30 min |
| 4. OpenBookQA | **nothing** — 0/25 defects and a structural explanation | — |

The κ gate matters more here than for the original plan, not less: "is this passage about this
question?" is a far more objective judgement than the five-way failure taxonomy, so κ should be
high, and if it is not, the rubric rather than the sample is at fault.

**What is deliberately *not* recommended.** Re-annotating `both_wrong` as a bucket. The previous
round established that 87.5% of the OBQA `both_wrong` items are model-side failures (extraction,
composition, distractor pull) rather than item defects, and this round establishes that the
item-defect component is small, enumerable, and detectable automatically. There is no remaining
question that reading the bucket would answer.

---

## 7. Limitations

1. ~~**Single annotator, no κ.**~~ **Resolved for SciQ (§8).** A second, human annotator censused
   the 33-item union screen blind; κ = 0.857 on the pre-registered 15-item subset, agreement 0.933.
   One rater is an LLM agent, so this is human-vs-LLM agreement — see §8.6 for what that does and
   does not license. The limitation stands unchanged for OpenBookQA and for every other hand label
   in this project.
2. **Recall above 0.40 is bounded by a sample, not a census.** 0/20 items checked above the
   threshold were mismatched; the 95% upper bound on that stratum is loose. The rate is therefore
   a **lower bound** — 1.47% as originally published, **1.36% after adjudication (§8)** — though
   §3.3's structural argument makes a large undercount unlikely.
3. **The OBQA estimate rests on 25 of 123.** A larger sample would tighten [0, 13.3%], but the
   structural argument in §2.3 does more work than the interval.
4. **Lexical only.** The detector cannot see a passage that is on topic but does not support the
   keyed answer — a semantic mismatch with high lexical overlap would pass undetected. Nothing in
   this audit addresses that class.

---

## 8. Adjudicated result — the double-annotated census (2026-09-09)

**This section supersedes §0's headline for SciQ.** §1–§7 are retained unchanged as the record of
the single-annotator round that produced the screen; where a figure there has moved, this section
says so and gives the new value.

### 8.1 Headline

| | |
|---|---:|
| **Confirmed topic mismatches (SciQ)** | **12** |
| **Corpus mismatch rate** | **12 / 884 = 1.36%** |
| Previously published (single annotator, §0) | 13 / 884 = 1.47% — **superseded** |
| Double-annotation n | **15** |
| Observed agreement | **0.933** |
| **Cohen's κ** | **0.857** |
| κ gate | **none** — `KAPPA_GATE = 0.70` is pre-registered for the Experiment-4 judge validation only |

Confirmed mismatch item ids: `23, 153, 203, 306, 412, 598, 602, 610, 655, 739, 746, 754`.

**OpenBookQA is unchanged at 0 / 500.** Step 4 of the §6 plan required nothing there.

### 8.2 Annotator configuration

| | |
|---|---|
| **Annotator 1** | the existing hand-check, `results/ex5_failure_audit/passage_overlap_handcheck.json` — produced in an earlier session **by an LLM agent**, single pass, before the rubric in `docs/guides/ANNOTATION_GUIDE.md` §2 existed |
| **Annotator 2** | a **human** annotator, labelling all 33 union-screen items from that rubric alone |
| Blinding | one-directional. Annotator 1's labels were fixed months earlier; annotator 2 worked from a blind copy (`corpus_defect_sheet_sciq33_BLIND.csv`) with `overlap`, `screen`, `gold_answer` and `row_id` removed, and did not read §3.1–3.2 or the hand-check artifact until the sheet was locked |
| Lock | `corpus_defect_sheet_sciq33_BLIND.LOCK.json`, SHA-256 `9cf39c47…1f1532`, recorded before any comparison |
| Comparison subset | pre-registered: `random.Random(20260904).sample(sorted(prior_labelled_ids), 15)` over the 21 items annotator 1 covered — fixed before labelling, in `corpus_defect_comparison_subset.json` |
| Scorer | `code/ex5_failure_audit/kappa_corpus_defect.py`; report in `corpus_defect_kappa.json` |

> ⚠ **One of the two raters is an LLM agent.** This κ is human-vs-LLM agreement on an objective
> question, not human-vs-human. It is nonetheless the **first inter-annotator κ of any kind in this
> project** — every other hand label here remains single-pass with no second rater (§7.1).

Confusion matrix over the 15 scored items:

```
                 human MISMATCH   human ON_TOPIC
  LLM MISMATCH          9                1
  LLM ON_TOPIC          0                5
```

### 8.3 Adjudication — item #584, resolved `ON_TOPIC`

One disagreement across the 15 scored items, and one across all 21 items the two raters shared
(agreement 20/21 = 0.952 on the wider set).

| | |
|---|---|
| Annotator 1 (LLM) | `MISMATCH` — *"Question is about the heart's atrium; the passage is about bronchi and respiratory bronchioles. Scores 0.286 only because 'respiratory'/'parts' coincide."* |
| Annotator 2 (human) | `ON_TOPIC` — *"The passage includes respiratory-system content but also directly discusses heart anatomy, including the atria and ventricles relevant to the question."* |
| **Adjudicated** | **`ON_TOPIC`** |

**Reason.** The passage is not limited to respiratory anatomy. It also directly discusses the
mammalian circulatory system and heart anatomy, explicitly including the atria and ventricles —
*"There is one atrium and one ventricle on the right side and one atrium and one ventricle on the
left side"* — which states the keyed answer outright. Under the specific-subject rubric
(`ANNOTATION_GUIDE.md` §2.3) it is on topic.

Annotator 1's recorded rationale characterises only the passage's opening two sentences. **This is
the class of error the double annotation was introduced to catch, and it was caught in the LLM
rater** — which is the substantive argument for keeping a human in this loop, independent of the κ
value.

Full record, including the unchanged κ inputs: `results/ex5_failure_audit/corpus_defect_adjudication.json`.

### 8.4 Figures elsewhere in this document that the adjudication moves

| Where | Was | Now |
|---|---|---|
| §0 headline, SciQ | 13 / 884 = 1.47% | **12 / 884 = 1.36%** |
| §3 hand-check, `topic_mismatch` | 13 (61.9% of the 21) | **12 (57.1% of the 21)** |
| §3.1 item list | includes #584 | **#584 removed** — adjudicated `ON_TOPIC` (§8.3) |
| §5, mismatches that are Setting-C ineligible | 12 of 13 (92.3%) | **12 of 12 (100%)** |
| §5, corpus base rate | 1.47% | **1.36%** |
| §5, H3 enrichment | 48.6× | **52.6×** (71.4% ÷ 1.36%) |
| §5, union of the two screens | "capturing 13 of 13 mismatches" | **"capturing 12 of 12 mismatches"** |

Unchanged: the screen sizes (21 overlap-only-or-both, 24 ineligible, 33 union); "Setting-C
ineligible items that are mismatches", still 12 of 24 (50.0%); OpenBookQA at 0/500; and §5's
qualitative conclusion — the two screens remain closely related, and their union is still a
complete detector at this scale. **The enrichment finding survives and strengthens slightly.**

### 8.5 New evidence: the ineligibility-only arm is clean

Twelve of the 33 union-screen items (`33, 54, 183, 237, 381, 386, 594, 639, 760, 762, 843, 844`)
entered through the Setting-C-ineligibility screen alone and had **never been hand-checked by
anyone**. Annotator 2 censused them: **zero topic mismatches**.

That confirms from the opposite direction what §5 argued — every confirmed mismatch is caught by
*both* screens, so the overlap screen alone would have sufficed. It also means all 12 confirmed
mismatches are Setting-C ineligible (100%, up from 92.3%), tightening the proxy relationship in §5.

### 8.6 Limitations

1. **One rater is an LLM agent** (§8.2). Human-vs-LLM κ, not human-vs-human.
2. **n = 15 for κ.** One disagreement moves it by roughly 0.06, so 0.857 should always be quoted
   with its n. The wider 21-item agreement (0.952) is descriptive only — it was not pre-registered.
3. **The rate remains a lower bound.** The census covers the 33-item union screen. Items above the
   0.40 overlap threshold were never censused — only a seeded 20-item sample of the 863, which
   returned 0 mismatches (§3.3). §7.2's reasoning is unchanged: a large undercount is structurally
   unlikely, but it is not excluded.
4. **κ covers the low-overlap items only.** Annotator 1 read only the 21 overlap-screen items, so
   the 12 ineligibility-only items (§8.5) have a single reading. The mismatch rate rests on all 33;
   κ speaks to 15 of them.
5. **The two annotators did not see identical fields.** Annotator 2's blind sheet omitted
   `gold_answer` by design, since the rubric does not use it (`corpus_defect_sheet_sciq33_BLIND_manifest.json`
   → `design_caveats`). A correctly applied rubric should be unaffected, but the asymmetry is real.
6. **§7's limitation 4 stands untouched:** the screen is lexical, so a passage that is on topic but
   does not support the keyed answer is invisible to it. Nothing here addresses that class.
