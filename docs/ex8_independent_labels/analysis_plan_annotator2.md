# Analysis plan for annotator 2 — fixed before labels

**Fixed 2026-09-22, before annotator 2 has returned any rating.** Annotator 1's results are already
known (`rq2_annotation_results.md`), so this plan is not blind to them. It is blind to annotator 2,
and it is written down now so that nothing below can be chosen after annotator 2's labels are seen.

The SHA-256 of this file is recorded in `results/ex8_independent_labels/analysis_plan_annotator2.LOCK.json`.
Committing both files before annotator 2's sheets come back would be the strongest evidence of
timing; a local hash alone can be regenerated along with the file.

**Decision taken:** option (a) + (c). Option (b) — averaging the raters into a new ground truth —
is not part of the plan (§6).

---

## 1. Inputs

| | Annotator 1 | Annotator 2 |
|---|---|---|
| Q1 | `q1_answerability_sheet.annotator1.csv` (filled, 100 rows) | `q1_answerability_sheet.annotator2.csv` (issued blank 2026-09-22, 100 rows) |
| Q2 | `q2_context_following_sheet.annotator1.csv` (filled, 55 rows) | `q2_context_following_sheet.annotator2.csv` (issued blank 2026-09-22, 55 rows) |

Annotator 2's blanks are byte-identical to the original issue: their SHA-256 matches
`annotation_sheets.LOCK.json`, and they were copied from the verified blanks, not derived from
annotator 1's files.

**Blindness conditions.** Annotator 2 receives the two blank sheets and `ANNOTATION_GUIDE_RQ2.md`
only — no repo access, no results doc, no manifest, no annotator 1 file, no LLM-judge output — and
works to the guide's protocol unchanged (Q1 and Q2 on separate days, no cross-referencing). Before
scoring, record whether annotator 2 had seen the results doc or any mentor update. If they had,
every annotator 2 result below is reported with that stated.

## 2. Structural validation — before any statistic

Identical to the checks run on annotator 1: row counts 100 / 55; columns and row order unchanged;
immutable columns unedited; every row either a valid value (Q1 `1`–`4`, Q2 `a`/`b`/`c`/`d`/`none`)
or marked `unusable` with the rating blank. Violations are reported with row numbers and **not
fixed**; a sheet that fails is returned to annotator 2 rather than repaired.

## 3. Primary analysis — unchanged

**Annotator 1's labels remain the primary ground truth.** The Part B results in
`rq2_annotation_results.md` stand as computed. Nothing about the primary sign test, AUC, Spearman,
binarisation (1–2 DEPENDENT / 3–4 NOT_DEPENDENT), strata, weights, or decision rule changes.

## 4. The κ gate — evaluated as pre-registered

**The gate is Cohen's κ on the collapsed binary over the pre-registered 30-item Q1 double block,
threshold 0.70** — exactly the statistic, subset and threshold fixed before any label existed.
Quadratic-weighted κ on the same 30 items is reported alongside. Q2 κ on the 16-item block is
reported without a gate, as pre-registered.

**Supplementary, not the gate:** the same statistics over the full sample — all 100 Q1 items, all 55
Q2 items. This is a stronger reliability estimate and it is reported, but it does not decide.

**If the block κ and the full-sample κ land on opposite sides of 0.70, the block κ governs** and the
disagreement is stated in the same sentence as the result. Neither figure is quoted without the
other.

**Gate outcome:**
- **Passes** → Part B is relabelled from exploratory to gated. The decision rule's branches become
  available, and the primary result is its INCONCLUSIVE branch as already computed.
- **Fails** → Part B stays provisional. Annotator 2's replication is still run and reported.

**Scorer change required.** `kappa_independent_labels.py` scores only `double_block_ids`. The
full-sample figure needs a scope option that scores every item both raters rated. The change is
limited to which ids are passed in; the block computation, the collapse rule, `KAPPA_GATE`, and the
unusable handling (an item either rater marks unusable is dropped from both) do not change. The
change is not yet made and must be made, and tested on synthetic sheets, before annotator 2's labels
are scored.

## 5. Replication — annotator 2 through the identical procedure

Annotator 2's full sheets are run through **unchanged** `compare_metrics.py` as a second, separate
label set:

```bash
python code/ex8_independent_labels/compare_metrics.py \
    --q1 results/ex8_independent_labels/q1_answerability_sheet.annotator2.csv \
    --q2 results/ex8_independent_labels/q2_context_following_sheet.annotator2.csv \
    --out results/ex8_independent_labels/metric_comparison.annotator2.json
```

No `--q1-b`: annotator 2 is a replication here, not a gate input — the gate is §4. Items annotator 2
marks `unusable` are dropped from annotator 2's run only.

### 5.1 What counts as replicated — Q1 sign test and metric comparison

**Annotator 1 is primary; annotator 2 either replicates it or does not.** A result is never
declared on annotator 2 alone.

| Primary (annotator 1) | Replication (annotator 2) | Reported as |
|---|---|---|
| INCONCLUSIVE (as computed: w/m 41/80, p 0.911) | INCONCLUSIVE | **Replicated.** No large difference between F and KDA_cont on Q1 |
| INCONCLUSIVE | any metric wins at p < 0.05 | **Not replicated.** Reported as annotator 2's result only. **No adoption.** |

The second row is the one this rule exists for: a significant result from the second rater does
not overturn the primary. For AUC and Spearman, report both raters' values and CIs side by side and
state for each metric whether annotator 2's CI excludes chance in the same direction as annotator
1's. The specific claim to check is §B2's: D is the only metric whose Q1 AUC CI excludes 0.5.

### 5.2 What counts as replicated — the Q2 findings

Three findings from the Q2 cross-tab (`rq2_annotation_results.md` §B4) are re-measured on annotator
2's Q2 sheet, with definitions fixed here:

- **Human reading of an item.** `reads CF target` if the chosen option index equals
  `counterfactual_target_idx`; `reads gold` if it equals `answer_idx`; `none` if `none`; `reads
  third option` otherwise. Indices come from the committed ex2 SciQ run.
- **Prior-override rate.** Among (model, question) pairs on items annotator 2 reads as the CF
  target, the share whose per-solver Setting-C class is `prior_dependent`, with a Wilson 95% CI.
  Annotator 1: 40/200 = 20.0% [15.0%, 26.1%].
- **Artifact share.** Among pair-level `prior_dependent` labels on the 55 items, the share on items
  annotator 2 does not read as the CF target, with a Wilson 95% CI. Annotator 1: 17/57 = 29.8%
  [19.5%, 42.7%]. The ensemble-level count (annotator 1: 5 of 16) is reported alongside.

For each finding, **consistent** means the two raters' 95% CIs overlap. Report both rates, both
CIs, and the verdict; never pool the counts. Also report item-level agreement between the two raters
on the four-way reading, and list the items where they differ.

The Q2 AUC comparison (annotator 1: F 0.964, KDA_cont 0.072, n = 5 negatives) is re-computed and
reported, but it is **not** a replication test — with a negative class this small, a change of one
or two items moves it arbitrarily.

**Script required.** These Q2 statistics were computed ad hoc in the previous round and are not in
`compare_metrics.py`. They are to be written as a script implementing exactly the definitions above,
checked against annotator 1's published figures (it must reproduce 40/200, 17/57, 5 of 16), and only
then run on annotator 2's sheet.

## 6. Not in the plan

- **Averaging the two raters, or taking Q2 by majority, as ground truth.** Not done. If it is ever
  run it is labelled post hoc, reported after the primary and the replication, and used for no
  decision.
- **Switching the primary to annotator 2**, or to whichever rater gives the clearer result.
- **Changing the binarisation cut, the strata, the weights, the thresholds, or the decision rule.**
- **Dropping items post hoc** other than those marked `unusable`.
- **Feeding the LLM-judge output (Part C) into anything above.** Its role is unchanged.

## 7. Reporting

Annotator 2's results go into `rq2_annotation_results.md` as a new part, structurally separate from
Part B (annotator 1, primary) and Part C (LLM judge), in this order: structural validation; κ gate
(block, then full sample); replication of Q1; replication of Q2; any deviation from this plan,
stated with the reason. A deviation is reported as a deviation, not written into the plan after the
fact.
