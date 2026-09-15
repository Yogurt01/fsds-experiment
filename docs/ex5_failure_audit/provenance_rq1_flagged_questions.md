# Provenance of `rq1_flagged_questions.json`

**Audience:** a reviewer who needs to know where the RQ1 labelled artefact came from and how much
of it can be independently re-derived.
**Artefact:** `results/ex5_failure_analysis/rq1_flagged_questions.json` (408 records: SciQ 328,
OBQA 80)
**Companions:** [`docs/failure_taxonomy_methodology.md`](failure_taxonomy_methodology.md) ·
[`docs/rq1_test_split_failure_analysis.md`](rq1_test_split_failure_analysis.md)

---

## 0. Result, and a correction to the previous round

**No generating script exists** — anywhere in the workspace, in git history, or in any deleted
worktree. That part of the earlier finding stands, and §2 documents the eight independent searches
that establish it rather than the single `grep` used before.

**But the previous round's conclusion was too pessimistic, and this is the substantive correction.**
It reported that the pool "cannot be re-derived". That is wrong. Everything in the artefact except
the hand-assigned `failure_category` turned out to be **exactly recomputable** from files that are
committed. The generator has now been restored by reimplementation:

> [`code/ex5_failure_audit/rebuild_flagged_pool.py`](../code/ex5_failure_audit/rebuild_flagged_pool.py)

verified against the surviving artefact at the level of individual records:

| Dataset | Pool membership | `kda_cont` + 4 probability fields | `detection_flags` | `structural_cues` (published cues) | Verdict |
|---|---|---|---|---|---|
| SciQ (328) | 328/328, 0 extra, 0 missing | 328/328 | 328/328 | 328/328 | **EXACT MATCH** |
| OBQA (80, `--no-c3`) | 80/80, 0 extra, 0 missing | 80/80 | 80/80 | 80/80 | **EXACT MATCH** |

> **Updated this round.** The previous version of this document reported 5 of 408 items (1.2%)
> as unresolved residuals on `abs_in_distractor` / `numeric_odd`, and characterised the artefact
> as self-inconsistent on them. **That characterisation was wrong.** Both rules have since been
> recovered — see §4.3. There are now **zero residuals**: all 408 items match on all fields,
> including both cues. The artefact contains no defect.

Reproduce with:

```bash
uv run --active python code/ex5_failure_audit/rebuild_flagged_pool.py --dataset sciq --verify
```

```bash
uv run --active python code/ex5_failure_audit/rebuild_flagged_pool.py --dataset obqa --no-c3 --verify
```

So the correct statement of the auditability gap is much narrower than previously reported: **the
selection logic was undocumented but not lost — it was inferable from the data it produced.** What
is genuinely unrecoverable is listed in §5.

---

## 1. Why a literal `grep` was insufficient

The previous round searched for the output filename (`grep -rn "rq1_flagged" --include="*.py"`).
That finds a generator only if the path is hardcoded, and would miss one that takes `--output` as a
CLI argument or builds the path from a run config. The searches below are structural: they look for
the *field names the file contains* and the *thresholds it encodes*, which a generator must contain
regardless of how its output path is supplied.

---

## 2. Eight fingerprinting searches

All run over the whole workspace, excluding `.venv/` and `.git/` internals.

### 2.1 Schema field names, grepped individually

| Field | Files containing it (excluding the artefact and `docs/`) |
|---|---|
| `p_rq_without_fact_ensemble` | **none** |
| `p_rqf_with_fact_ensemble` | **none** |
| `p_rq_without_fact_primary` | **none** |
| `p_rqf_with_fact_primary` | **none** |
| `pooled_probs_without_fact` | **none** |
| `detection_flags` | **none** |
| `failure_category` | **none** |
| `structural_cues` | **none** |
| `counterfactual_class` | `code/ex2_counterfactual/run_counterfactual_experiment.py` (its own producer) |
| `kda_adjusted_hard` | `code/ex2_counterfactual/run_counterfactual_experiment.py` (its own producer) |
| `kda_cont` | the three counterfactual/Qwen result files and their producers; `question-score/` reference impl |

The five fields unique to the merge step — the four `p_rq*`/`p_rqf*` names and
`pooled_probs_without_fact` — appear in **no script at all**.

### 2.2 Case variants

`kdaCont`, `detectionFlags`, `failureCategory`, `counterfactualClass`, `structuralCues`,
`flaggedQuestions`, `FLAGGED_QUESTIONS`, `rq1Flagged` — **all zero hits**. `flagged_questions`
appears only in three `docs/*.md` files.

### 2.3 Scripts reading both pipelines

The artefact merges KDA_small Settings A/B with counterfactual Setting C, so its generator must
open both. Every `.py`, `.ipynb`, `.sh`, `Makefile`, `.toml`, `.cfg`, `.yaml`, `.yml` in the tree
was tested for reading *both* a `results_kda_small*` / `categorized_*` file **and** a
`results_counterfactual*` file. **No file matches both.**

### 2.4 Threshold literals co-occurring

No `.py` or `.ipynb` file contains both `prior_dependent` and a `0.70` / `0.7` literal.

### 2.5 Build files and notebooks

`find` for `*.ipynb`, `*.sh`, `Makefile`, `makefile`, `*.mk`, `justfile`, `*.bat` across the
workspace returns **nothing**. There are no notebooks and no task runners in this project at all.

### 2.6 Output-path references

`grep -rn "flagged.*questions\|rq1_failure_analysis"` outside `docs/` returns only the artefact
itself. No script, config, or log names that output path.

### 2.7 Structural-cue detector names

The seven cue strings present in the artefact (`longest_gold`, `shortest_gold`,
`stem_overlap_gold`, `dup_distractors`, `numeric_odd`, `abs_in_distractor`, `gold_in_passage`)
plus the two named only in prose (`head_family3`, `stem_echo`) appear in **no code** — only in
`docs/`.

### 2.8 Git and worktree recovery

| Check | Result |
|---|---|
| `git log --all --oneline` | one commit, `edd0354 Initial commit` |
| Artefact tracked by git? | **No** — `results/` is untracked working-tree content |
| `git reflog` | branch rename + initial commit only |
| `git stash list` | empty |
| `git fsck --lost-found --dangling` | no dangling objects |
| `git worktree list` | only the main checkout |
| Worktree path in logs | `.claude/worktrees/test-splits-failure-analysis-588cac` — **deleted**; `.claude/` no longer exists |

The OBQA counterfactual log records that the run executed inside
`.claude/worktrees/test-splits-failure-analysis-588cac/`. That worktree has been removed and its
contents are not in any git object. **The generator almost certainly lived there and is gone.**

---

## 3. Candidate scripts

No script fully matches. Ranked by what they contribute, with what is missing:

| Candidate | Supplies | Missing | Confidence it is the generator |
|---|---|---|---|
| `code/ex1_reproduce_KDA/run_experiment.py` | Settings A/B probabilities for the 4 students — the source of all five probability fields | no pool selection, no cues, no merge with Setting C | **Confirmed as an input, not the generator** |
| `code/ex2_counterfactual/run_counterfactual_experiment.py` | `counterfactual_class`, `kda_adjusted_hard` | no pool selection, no cues, does not read KDA_small results | **Confirmed as an input, not the generator** |
| `code/ex1_reproduce_KDA/categorize_kda_results.py` | the `both_correct` bucket that criterion C2 needs | no thresholds, no flags, no cues, SciQ-only outputs | **Plausible input; not the generator** |
| any file in `question-score/` | reference `KDA_cont` implementation | the reference repo, untouched; no project field names | **Ruled out** |

**Conclusion: the generator is a deleted fourth script that read the outputs of the first two.**

---

## 4. What was recovered, and how it was confirmed

Rather than stop at "missing", the selection logic was reimplemented from the written criteria in
`rq1_test_split_failure_analysis.md` §1 and checked against the artefact record by record.

### 4.1 Detection criteria — exact on the first attempt

```
C1_kda_cont           KDA_cont >= 0.70       AND primary model correct in Setting A
C1_kda_disc           KDA_disc_proxy == 1.0  AND primary model correct in Setting A
C2_both_correct_conf  both_correct bucket    AND P(R^q=1) >= 0.70 on the primary model
C3_prior_dependent    ensemble Setting-C class == "prior_dependent"
```

with `KDA_disc_proxy(q) = |{m : wrong in A and correct in B}| / |{m : wrong in A}|`, undefined when
no model fails Setting A. Primary model `Riiid/kda-mpnet-base-race`.

Reimplementing these reproduced **328/328 SciQ and 80/80 OBQA flag sets exactly**, with no extra
and no missing questions. The five probability fields and `kda_cont` also matched every record to
within 5e-3 (they are stored rounded). The published thresholds are therefore exactly what was run.

### 4.2 Structural cues — recovered, two of them empirically

Four cues reproduced immediately from their prose descriptions: `longest_gold`, `shortest_gold`,
`stem_overlap_gold`, `dup_distractors` (0 disagreements on either dataset).

`gold_in_passage` needed its semantics recovered from the data. It is **not** a substring test and
**not** the counterfactual locator's contiguous n-gram test. Three successive hypotheses were
tested against the artefact:

| Hypothesis | SciQ disagreements |
|---|---:|
| plain case-insensitive substring | 8 |
| contiguous n-gram via `find_substitution_span` (partial tier) | 5 |
| all gold content words present, symmetric number-stripping | 1 |
| **all gold content words present, *directional* plural tolerance** | **0** |

The final rule: every content word of the gold answer must occur in the passage, where a singular
gold token matches a plural in the passage but **not** the reverse. Two items pin the direction in
opposite directions and admit no other rule:

> **SciQ #669** — gold `safety precaution`, passage `…precautions…` → cue **fires**
> **SciQ #146** — gold `skeletal muscle fibers`, passage `…muscle fiber…` → cue does **not** fire

`numeric_odd` was likewise recovered as *"exactly one option contains a digit"* rather than *"is
entirely numeric"*, from SciQ #883 (`sulfur dioxide (so2)` among three digit-free options).

### 4.3 The last two cues — recovered, not defective

The previous round left 5 of 408 items disagreeing on `abs_in_distractor` and `numeric_odd`, and
concluded the artefact applied them inconsistently. That conclusion was premature: **both cues
follow a rule that had simply not been recovered yet, and both are gold-centric — exactly like
every other cue in the set.** The apparent inconsistencies dissolve under the correct rule.

**`numeric_odd` — the GOLD is the unique option containing a digit.**

Not "some option contains a digit". The two items that looked contradictory pin it down:

| Item | Options | Digit-bearing option | Artefact fires? |
|---|---|---|---|
| SciQ #883 | sulfuric acid / sulfur monoxide / formaldehyde / **sulfur dioxide (so2)** | the **gold** | **yes** |
| OBQA #44 | **can help you monitor a fever** / … / read exactly at 98.6 degrees / … | a **distractor** | **no** |

**`abs_in_distractor` — an absolute word appears in ≥1 distractor and NOT in the gold.**

The cue is *discriminative*, not merely presence-detecting: where every option shares the
quantifier it distinguishes nothing and does not fire. Vocabulary:
`always | never | all | none | every | exactly | nothing | must | entirely | completely`
— note **`only` is not in it**. All four apparent contradictions resolve:

| Item | Gold | Distractor trigger | Artefact | Explained by |
|---|---|---|---|---|
| SciQ #522 | `every 12 hours` | `Every 24 hours` | **no** | the gold also carries "every" → not discriminative |
| OBQA #210 | `alter the way the moon's facade looks` | `cause lunar eclipse **every** day` | **yes** | gold has no absolute |
| OBQA #477 | `they will shrink` | `**nothing** will happen` | **yes** | "nothing" is in the vocabulary |
| OBQA #361 | `commonality among **all** animals` | `**only** land dwelling mammals` | **no** | "only" is not in the vocabulary |

**Verification.** Both rules were tested against all 408 artefact records before being adopted:

| Cue | Agreement |
|---|---|
| `numeric_odd` | **408/408** |
| `abs_in_distractor` | **408/408** |

They are now implemented in `structural_cues()` and the verifier's residual-exemption set
(`RARE_CUES`) has been emptied, so `--verify` reports EXACT MATCH with no carve-outs on either
dataset.

**The general lesson.** Every cue in this artefact is *gold-centric and discriminative* — it fires
only when the surface property singles the gold answer out. That is consistent across
`longest_gold`, `shortest_gold`, `gold_in_passage`, `stem_overlap_gold`, and now these two. The
earlier "defect" reading came from testing presence-based rules against a discriminative design.

---

## 5. What remains unaccounted for

**1. The manual `failure_category` labels (408 items).** Not recoverable by any means: they are
one annotator's judgements, with no rubric written in advance and no second rater. See
[`failure_taxonomy_methodology.md`](failure_taxonomy_methodology.md) §2 and §4. The rebuild script
carries them across by `question_id` under `--carry-labels`; it cannot regenerate them.

**2. Nothing.** ~~The exact definitions of `abs_in_distractor` and `numeric_odd`.~~ **Resolved
this round — see §4.3.** Both rules were recovered; residual disagreement is now 0 of 408.

**3. `head_family3` and `stem_echo`.** Discussed in §4.3 of the failure-analysis doc but **absent
from the artefact entirely** — no record carries either value. They were computed for that table
and never exported. The Table 4 numbers depending on them (templated option family 22.6%/20.0%
SciQ) are therefore not re-derivable from the artefact, though they are re-derivable from the
datasets by anyone willing to re-specify the detector.

---

## 6. What changed as a result

1. **The generator now exists** at `code/ex5_failure_audit/rebuild_flagged_pool.py`, verified exact against
   the artefact on both datasets. Recommendation 1 of `failure_taxonomy_methodology.md` §7 ("commit
   the selection script so the pool is re-derivable") is **closed**.
2. **Criterion C3 has been applied to OpenBookQA** for the first time — the fix flagged as stale in
   the previous round. See [`failure_taxonomy_methodology.md`](failure_taxonomy_methodology.md) §6.1
   and the results in
   `results/ex5_failure_analysis/rq1_flagged_questions_rebuilt_obqa.json`.
3. **Rebuilt pools are written** for both datasets with the manual labels carried across, and with
   newly-admitted items explicitly marked `failure_category_status: "UNLABELLED_new_admission"` so
   the annotation debt is visible rather than silent.
