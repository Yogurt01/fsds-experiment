# Benchmark-Wide Corpus-Defect Audit — Question/Passage Overlap

**New this round.** Nothing in this document was reported previously; it supersedes the *scoping
note* that proposed this screen in
[`unexploited_buckets_analysis.md`](unexploited_buckets_analysis.md) §6.

| | |
|---|---|
| Scope | **All 1,384 test items** — SciQ 884 + OpenBookQA 500 — independent of any model's performance |
| Script | [`code/ex5_failure_audit/passage_overlap_audit.py`](../code/ex5_failure_audit/passage_overlap_audit.py) |
| Hand-check | `results/ex5_failure_audit/passage_overlap_handcheck.json` |
| Distribution data | `results/ex5_failure_audit/passage_overlap_audit.json` |

```bash
uv run --active python code/ex5_failure_audit/passage_overlap_audit.py
```

---

## 0. Headline

| Dataset | Confirmed topic mismatches | Rate | Basis |
|---|---:|---:|---|
| **SciQ** | **13 / 884** | **1.47%** | census of all 21 items below 0.40 overlap + 20-item sample above |
| **OpenBookQA** | **0 / 500** | **0%** (95% CI upper ≈ 3.2% of split) | 25-item sample of the 123 zero-overlap items |

Three results that change how the earlier findings should be read:

1. **SciQ has a real, small, fully enumerable corpus defect**: 13 items whose support passage is
   about an unrelated subject. All 13 are detected by the screen.
2. **OpenBookQA has no such defect at all**, and its zero-overlap mass — 24.6% of the split — is
   entirely by design. The detector does not transfer between the two benchmarks, for the same
   structural reason the counterfactual mechanism did not.
3. **The earlier SciQ H3 finding (5 of 7) is not in line with the base rate — it is enriched
   48.6×** over it. See §5.

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
([`counterfactual_experiment_methodology.md`](counterfactual_experiment_methodology.md) §5.1)
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

| Category | n | Share of the 21 examined |
|---|---:|---:|
| `topic_mismatch` | **13** | 61.9% |
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
| **#584** | the heart's atrium | bronchi and respiratory bronchioles |
| **#602** | how the water cycle ends | domestic hot-water and warm-air heating |

**#584 and #602 are the important ones**: both sit *above* the 0.10 threshold (0.286 and 0.333)
and would have been missed by it. They score non-zero only on incidental words — "respiratory"
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

| | Value |
|---|---:|
| Corpus base rate of topic mismatch (SciQ) | **1.47%** |
| Rate inside the H3 screen pool (`both_wrong` ∧ Setting-C-ineligible, n=7) | **71.4%** |
| **Enrichment** | **48.6×** |

The two screens are closely related, which explains the size of the effect and should be stated
plainly rather than treated as an independent confirmation — a passage about the wrong topic will
not contain the gold answer, so it is automatically Setting-C-ineligible:

| Relationship | Value |
|---|---|
| Mismatches that are Setting-C ineligible | **12 of 13 (92.3%)** |
| Setting-C ineligible items that are mismatches | 12 of 24 (50.0%) |
| Items with overlap < 0.40 that are mismatches | 13 of 21 (61.9%) |
| **Union of the two screens** | **33 unique items, capturing 13 of 13 mismatches** |

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

1. **Single annotator, no κ.** Same standing limitation as every hand label in this project. The
   topic-mismatch judgements are unusually clear-cut (a fungi passage under an osteoporosis
   question), but they have not been independently confirmed.
2. **Recall above 0.40 is bounded by a sample, not a census.** 0/20 items checked above the
   threshold were mismatched; the 95% upper bound on that stratum is loose. The 1.47% figure is
   therefore a **lower bound**, though §3.3's structural argument makes a large undercount
   unlikely.
3. **The OBQA estimate rests on 25 of 123.** A larger sample would tighten [0, 13.3%], but the
   structural argument in §2.3 does more work than the interval.
4. **Lexical only.** The detector cannot see a passage that is on topic but does not support the
   keyed answer — a semantic mismatch with high lexical overlap would pass undetected. Nothing in
   this audit addresses that class.
