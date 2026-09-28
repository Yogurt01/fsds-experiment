# Residual-leakage rescan of the SciQ counterfactual passages, and the corrected prior-dependence figures

**Run 2026-09-24.** Read-only with respect to every committed run. Nothing in ex2, ex6 or
ex8 was modified, re-run or reweighted; this is a second measurement over the same
committed SciQ Setting-C output.

Scripts: [`rescan_residual_leakage.py`](../../code/ex7_counterfactual_construction/rescan_residual_leakage.py),
[`recompute_prior_dependence.py`](../../code/ex7_counterfactual_construction/recompute_prior_dependence.py).
Outputs: `results/ex7_counterfactual_construction/residual_leakage_rescan_sciq.json`,
`results/ex7_counterfactual_construction/prior_dependence_corrected_sciq.json`.

---

## 1. What was wrong with the original check

ex2 records one leakage counter per item, `residual_answer_mentions`: a **whole-word,
exact-surface** count of the gold answer in the perturbed passage. It reads **0 on all 860
eligible SciQ items**, which ex7 reported as zero whole-phrase leakage.

That zero is close to tautological. The substituter has just rewritten every whole-word
occurrence of the matched surface form, so a whole-word residual of the *same* surface form
is near-impossible by construction. The check cannot see:

- an inflected form the substituter did not touch (`immune system` → `immune systems`),
- the gold fused to the preceding word (`are calledoxidants`),
- a clause elsewhere in the passage that still states the same fact after only a fragment
  of the gold was rewritten.

All five ex8 Q2 construction failures — items 0, 25, 222, 716, 805, where a human reading
the counterfactual passage still answered **gold** — read `residual_answer_mentions = 0`.
ex2's secondary `residual_glued_mentions` counter fires on 16 items and catches only two of
the five (0 and 222).

## 2. The expanded check

Four matchers plus a sentence-level filter. Matches that fall inside the inserted
counterfactual target are never counted (the target `similar state of solute` contains the
matched fragment `of solute`; crediting that would be an artefact of the rewriter).

| | Matcher | Catches |
|---|---|---|
| R1 | exact whole word | ex2's original check, reproduced for comparability |
| R2 | gold fused to the preceding word | `calledoxidants` |
| R3 | morphological / stem variant as a whole word | `acids`, `immune systems`, `antioxidant` |
| R4 | a sentence carrying all but at most one of gold's content stems, retaining ≥ 2 | the item-222 fragment-splice pattern |
| Q | the carrying sentence shares ≥ 3 of the question's content stems, or ≥ 50% of them | "is this still the sentence a reader would answer from" |

- **`flagged_any` = R1 ∨ R2 ∨ R3 ∨ R4** — the gold is still somewhere in the passage.
- **`flagged_strict` = `flagged_any` ∧ Q** — the gold is still stated *where the question is
  answered*. This is the headline.

Stemming is a deterministic suffix stripper written for this check (no NLP dependency, in
keeping with the rest of the construction code); R3 also runs ex2's own
`morphological_variants` list, and the two are reported separately.

**One threshold was changed after the precision check in §4** and is flagged as such:
`FRAGMENT_MIN_PRESENT_STEMS = 2`. Without it, a two-word gold fires on its generic head
noun alone — `cells` for `voltaic cells`, `light` for `visible light` — which is a common
noun surviving, not the gold answer surviving. Everything else was fixed from the five
known failure patterns before the 860-item run.

## 3. Result over all 860 eligible SciQ items

| Check | Items flagged | Rate |
|---|---:|---:|
| ex2 `residual_answer_mentions` (published) | **0** | 0.0% |
| ex2 `residual_glued_mentions` | 16 | 1.9% |
| expanded, `flagged_any` | **109** | 12.7% |
| expanded, `flagged_strict` | **30** | 3.5% |

25 of the 30 strict flags are invisible to *both* ex2 counters.

Which matcher fires (an item can trigger more than one):

| Matcher | in `flagged_strict` | in `flagged_any` |
|---|---:|---:|
| morphological (variant list) | 20 | 73 |
| morphological (stem equality) | 1 | 2 |
| fragment | 12 | 42 |
| glued | 3 | 9 |
| exact | 0 | 0 |

The morphological matcher is the single largest source, as the five known failures predicted.

### By substitution tier

| Tier | n | `flagged_any` | `flagged_strict` | strict rate |
|---|---:|---:|---:|---:|
| exact | 813 | 102 | 26 | 3.2% |
| partial | 36 | 6 | 3 | 8.3% |
| morphological | 11 | 1 | 1 | 9.1% |
| glued | 0 | — | — | — |

The two lower tiers are roughly 2.5× as likely to leak as `exact`, which is the expected
direction, but they hold only 47 of the 860 items, so 26 of the 30 strict flags sit in the
tier the project treats as cleanest. **Restricting to `exact` would not have avoided this.**

## 4. Validation against the 55-item Q2 sample

The Q2 sheet is the only part of SciQ with human ground truth on what the counterfactual
passage says. Annotator 1 read 5 of 55 items as stating the **gold** answer and 50 as
stating something else.

| | Result |
|---|---|
| Strict recall on the 5 known failures | **5 / 5** (items 0, 25, 222, 716, 805) |
| Strict flags among the 50 others | **2 / 50** (items 146, 774) |

Both discrepancies, as required:

- **Item 774 is a genuine leak the human did not act on.** Gold `sebaceous gland` → target
  `melanin gland`. The rewriter changed the answering sentence but left the section heading
  `Sebaceous Glands` immediately in front of it, and left `Most sebaceous glands are
  associated with hair follicles.` intact further down. The annotator read the target,
  correctly, because the answering sentence does state the target — but the gold term
  plainly survives. This is the detector being right and the item being ambiguous, not a
  false positive.
- **Item 146 is a real limitation.** Gold `skeletal muscle fibers` → target `heart muscle
  fibers`. The passage repeatedly says `muscle fiber` without either modifier, so R4 fires
  on `muscle` + `fiber`. Item 222 (a true failure) has the identical shape: a three-stem
  gold with two stems surviving. **No lexical rule separates 146 from 222**, so this class
  of false positive is not removable within the current approach.

Recall is 5/5 by design — the matchers were built from these five patterns, which is what
was asked for. The 50-item side is the informative half, and it is the one that produced
the `FRAGMENT_MIN_PRESENT_STEMS` correction in §2.

### Hand adjudication of the 30 strict flags

Recorded in `ADJUDICATION` in `recompute_prior_dependence.py` so the conservative bound is
reproducible. **This is my own single reading, not an independent human annotation, and is
not to be reported as one.** The only rater-verified items in it are the five the ex8
annotators reached independently.

| Verdict | n | Meaning |
|---|---:|---|
| CONFIRMED | **16** | gold still stands in the answering sentence or an equivalent clause |
| BORDERLINE | 9 | gold survives in a topical or adjacent clause; a reader's response is genuinely unclear |
| SPURIOUS | 5 | question-stem echo (40, 57), generic head noun (146, 209), or gold buried in a longer word (752) |

## 5. Corrected prior-dependence figures

`recompute_prior_dependence.py` calls ex2's own `compute_model_metrics`, so the corrected
numbers come from exactly the code that produced the published ones. With no exclusions it
reproduces every committed per-model figure exactly (asserted in the script).

Three exclusion sets: **confirmed** (16, conservative), **strict** (30, primary), **any**
(109, upper bound).

### 5.1 Aggregate movement is small

`prior_dependent` share of all eligible items, in per cent:

| Solver | published (n=860) | −confirmed (844) | **−strict (830)** | −any (751) | Δ strict |
|---|---:|---:|---:|---:|---:|
| t5-small-ssm-nq | 32.79 | 31.99 | **31.81** | 30.76 | −0.98 |
| albert-xlarge | 19.30 | 18.48 | **18.31** | 16.78 | −0.99 |
| mpnet-base | 8.26 | 6.99 | **6.99** | 7.06 | −1.27 |
| scibert | 16.63 | 16.23 | **16.14** | 15.31 | −0.48 |
| **ensemble** | **13.37** | 12.44 | **12.17** | 11.58 | **−1.20** |

Context-verified accuracy with fact, in per cent:

| Solver | published | **−strict** | Δ |
|---|---:|---:|---:|
| t5-small-ssm-nq | 39.42 | **38.92** | −0.50 |
| albert-xlarge | 56.63 | **55.90** | −0.72 |
| mpnet-base | 85.81 | **86.51** | +0.69 |
| scibert | 58.72 | **58.55** | −0.17 |
| **ensemble** | **78.49** | **78.67** | **+0.19** |

Prior inflation of accuracy-with-fact moves by −0.70 to +0.15 points; the ensemble figure
goes 9.88% → 9.52%. Context-following rate (`counterfactual_passage_vs_cf_target`) rises
between +0.29 and +1.28 points, ensemble 77.79% → 78.92%, which is the expected direction:
removing items where the passage still says gold makes the remaining passages easier to
follow.

**No published conclusion changes.** Every solver keeps its rank, and the ensemble
prior-dependence rate stays in double digits.

### 5.2 But the mechanism is confirmed, and it is strong

The aggregate barely moves because the flagged set is 3.5% of the corpus. The rate *within*
the flagged set is the actual evidence:

`prior_dependent` rate among flagged vs unflagged items, with Wilson 95% intervals:

| Solver | flagged_strict (n=30) | adjudicated confirmed (n=16) | unflagged (n=751) |
|---|---|---|---|
| t5-small-ssm-nq | 60.0% [42.3, 75.4] | 75.0% [50.5, 89.8] | 30.8% [27.6, 34.2] |
| albert-xlarge | 46.7% [30.2, 63.9] | 62.5% [38.6, 81.5] | 16.8% [14.3, 19.6] |
| mpnet-base | 43.3% [27.4, 60.8] | 75.0% [50.5, 89.8] | 7.1% [5.4, 9.1] |
| scibert | 30.0% [16.7, 47.9] | 37.5% [18.5, 61.4] | 15.3% [12.9, 18.1] |
| **ensemble** | **46.7% [30.2, 63.9]** | **62.5% [38.6, 81.5]** | **11.6% [9.5, 14.1]** |

At the ensemble level a flagged item is **4.0×** as likely to be called `prior_dependent` as
an unflagged one, and an adjudicated-confirmed leak is **5.4×** as likely; the intervals do
not overlap for four of the five solvers. On mpnet the ratio is 10.6×.

So the ex8 Q2 finding generalises: **where the construction leaks, the Setting-C label is
systematically wrong in the `prior_dependent` direction.** What it does *not* do is rewrite
the corpus-level numbers, because the leak is rare.

## 6. What this does and does not settle

- **Settled.** ex2's `residual_answer_mentions` check is not a meaningful leakage measure;
  its 0/860 reading is an artefact of matching the same surface form the rewriter just
  replaced. The expanded check finds 30 items (3.5%) where the gold is still stated in the
  answering position, 25 of them invisible to both existing counters.
- **Settled.** Those items carry `prior_dependent` at 4–5× the base rate, so the defect
  does manufacture prior-dependence labels, exactly as the five Q2 failures suggested.
- **Settled.** Corpus-level SciQ prior-dependence figures move by about one point and no
  published conclusion changes.
- **Not settled.** The 30 is a lower bound on a lexical check. Item 222's failure is
  semantic (`chemical state of solute` restated as `chemical identity of the solute`) and
  was caught only because enough stems happened to survive; a paraphrase with no shared
  stems would pass. Nothing here estimates how many of those exist.
- **Not settled.** The `BORDERLINE`/`CONFIRMED` split is my reading alone. Adjudicating the
  30 with a human annotator would tighten §5.2's conservative bound; it is not needed for
  §5.1.
- **Not run.** The same rescan on OBQA. OBQA was closed out in §11 of the construction
  audit and this pass did not reopen it.
