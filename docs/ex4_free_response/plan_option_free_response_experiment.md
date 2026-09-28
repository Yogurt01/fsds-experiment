# Plan — Option-Free Free-Response Experiment

**Status: harness built; two pilots run (forward-only, then bidirectional). Full run BLOCKED.**
The plan is in §1–§7; §8 reports the first pilot and revises the plan; **§9 reports the
bidirectional pilot and the validation tooling built this round.**

> **The full run is blocked on two conditions, neither of which can be cleared locally:**
> 1. **Human validation gate** — 150 items hand-labelled, Cohen's κ ≥ 0.70 against the judge
>    (§9.3). Tooling is built and ready; the labelling is the user's.
> 2. **Qwen2.5-7B cross-judge comparison** — queued as entry 3 in
>    [`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md), to be run on Colab/Kaggle.
>
> Do not launch the full run until both clear.
**Audience:** a reviewer verifying that the experiment does not yet exist, and assessing the
proposed design.
**Companions:** [`docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md`](../ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md) ·
[`docs/ex5_failure_audit/rq1_test_split_failure_analysis.md`](../ex5_failure_audit/rq1_test_split_failure_analysis.md) ·
[`docs/ex3_student_simulation/student_persona_simulation_report.md`](../ex3_student_simulation/student_persona_simulation_report.md)

---

## 0. Confirmation that it has not been run

Asked directly: *has any experiment stripped the answer options, prompted the model to answer in
free text, and matched that text against the gold answer string?*

**No.** Three independent checks:

1. **No generation code exists anywhere in the repository.**
   ```bash
   grep -rn "\.generate(\|max_new_tokens\|GenerationConfig\|num_beams\|do_sample" code/
   # → no matches
   ```
   All 19 Python files under `code/` are listed in §0.1. Every scoring path is
   **closed-set**: either a softmax over four multiple-choice logits
   ([`kda_tiny.py:161-178`](../../code/ex1_reproduce_KDA/kda_tiny.py)), a length-normalised NLL over
   the four option strings (`kda_tiny.py:238-264`), or a softmax over the four option *letters*
   at one position ([`kda_qwen_eval.py`](../../code/ex1_reproduce_KDA/kda_qwen_eval.py),
   `score_prompt`). Not one of them can emit a token the option list did not supply.

2. **No results file contains free-text answers.** Scanning every JSON under `results/` for keys
   such as `generated`, `free_text`, `free_response`, `model_answer_text`, `open_ended` returns
   nothing.

3. **The "no-fact" variant that does exist is still multiple-choice.** Setting A
   (`accuracy_without_fact`, `P(R^q=1)`) removes the *passage*, never the *options*. Every
   zero-context number in every report — SciQ 0.954, OBQA 0.826 for Qwen3-4B — is
   4-way-multiple-choice accuracy with a 0.25 chance floor.

### 0.1 Every scoring path in the repository

| Script | Scoring mode | Closed-set? |
|---|---|---|
| `ex1_reproduce_KDA/run_experiment.py` | 4-model ensemble, MC head / T5 option NLL | Yes |
| `ex1_reproduce_KDA/kda_tiny.py` | `AutoModelForMultipleChoice` softmax; T5 option NLL | Yes |
| `ex1_reproduce_KDA/kda_qwen_eval.py` | letter-logit softmax over A/B/C/D | Yes |
| `ex1_reproduce_KDA/run_qwen7b_eval.py` | letter-logit softmax over A/B/C/D | Yes |
| `ex2_counterfactual/run_counterfactual_experiment.py` | reuses `kda_tiny` students | Yes |
| `ex3_student_simulation/run_student_simulation.py` | letter-logit softmax over A/B/C/D | Yes |

**Why this matters.** The headline motivating fact of the whole programme — "Qwen3-4B is
parametrically saturated at 95.4% zero-context on SciQ" — is a claim about *elimination-based
multiple choice*. It is not yet known whether the model can **recall** those answers unprompted.
The two are different capabilities, and the gap between them is exactly what this experiment
measures.

---

## 1. Research question

> Does open-ended recall diverge from MCQ-format zero-context accuracy? I.e. how much of the 95.4%
> SciQ / 82.6% OBQA zero-context accuracy is genuine parametric knowledge, and how much is
> recognition and elimination that the option list makes possible?

Sub-question: does the divergence differ between the two datasets, given that SciQ gold answers
are 63% single words drawn from textbook prose while OBQA gold answers are 3.26-word phrases
written by annotators?

---

## 2. Conditions

Four cells, of which two are new runs and two already exist.

| Cell | Options | Fact/passage | Status | Measures |
|---|---|---|---|---|
| **A′** | **removed** | removed | **NEW** | open-ended parametric recall |
| **B′** | **removed** | present | **NEW** | open-ended reading comprehension |
| A | present | removed | exists (ex1) | MCQ zero-context (0.954 / 0.826) |
| B | present | present | exists (ex1) | MCQ with fact (0.998 / 0.920) |

**A′ is the primary cell.** B′ is included because it is nearly free once the harness exists and
it separates two failure causes: an item failed in A′ *because the model does not know it* versus
*because the expected answer string is unguessable without seeing the options* (severe on OBQA —
see §4.3). Without B′, a low A′ score cannot be attributed.

The key comparison is **A − A′** per dataset: the accuracy attributable to having the options.

---

## 3. Prompt templates

Deliberately minimal, mirroring the existing style
(`kda_qwen_eval.py`, `SYSTEM_PROMPT` / `build_prompt`) so the only change from the published runs
is the removal of the option list.

**Cell A′ (no options, no fact):**

```
System: You are a student taking a short-answer science exam. Answer with the
answer only - a word or short phrase. Do not explain, do not write a sentence.

User: Question: {question}

Answer with the answer only (a word or short phrase).
```

**Cell B′ (no options, fact present):**

```
System: You are a student taking a short-answer science exam. Answer using the
reference text. Answer with the answer only - a word or short phrase. Do not
explain, do not write a sentence.

User: Fact: {passage}

Question: {question}

Answer with the answer only (a word or short phrase).
```

**Decoding:** greedy (`do_sample=False`), `max_new_tokens=24`, stop at newline. Greedy keeps the
run deterministic and comparable to the existing argmax-based settings. 24 tokens comfortably
exceeds the longest gold answer (SciQ 6 words, OBQA 15 words) while capping runaway generations.

**Post-processing before matching:** strip a leading `Answer:` / `A:`, take the first line, strip
surrounding quotes and a trailing period. Recorded verbatim alongside the raw generation so every
transformation is auditable.

---

## 4. Matching methodology

### 4.1 The decision

**A three-stage cascade: normalised exact match → alias/morphological match → LLM-judge
adjudication of the residual only.** Every item's stage is recorded, so any headline number can be
reported with and without the judged tier.

### 4.2 Why, and why not the alternatives

| Option | Verdict |
|---|---|
| **Exact string match alone** | Rejected. Fails on `oxidant`/`oxidants`, `tree ring`/`tree rings`, and every article difference (`a marsh` vs `marsh`). Would understate recall, and unevenly across datasets. |
| **Embedding similarity with a threshold** | Rejected. No principled threshold exists, and the failure mode is severe here: on a 4-option science item, `mitosis` and `meiosis` are close neighbours in every general-purpose embedding space while being opposite answers. A threshold tuned to accept `tree ring`≈`tree rings` will also accept near-miss distractors. It would also add a model dependency with no audit trail. |
| **LLM-judge on every item** | Rejected as the *primary* mechanism. It is the most expensive option, makes the headline number a function of an unaudited second model, and is wasted on the 63% of SciQ items that are single words where deterministic matching is exact and free. |
| **Cascade (chosen)** | The deterministic stages resolve the large, easy majority with a fully auditable rule; the judge is confined to the genuinely ambiguous residual, which is small enough to human-spot-check in full. |

### 4.3 The dataset asymmetry that forces a cascade

Measured from the prepared splits:

| | SciQ (884) | OBQA (500) |
|---|---:|---:|
| Gold answer mean word count | **1.48** | **3.26** |
| Median | 1 | 3 |
| Maximum | 6 | 15 |
| Single-word golds | **63.1%** | 31.4% |
| ≤ 3 words | **99.0%** | 62.8% |
| Gold appears verbatim in the reference text | **92.8%** | **9.4%** |
| Golds containing punctuation/symbols | 7 | 23 |

Examples — SciQ: `oxidants`, `clone`, `backbone`, `elevation`, `tree rings`.
OBQA: `quit eating lunch out`, `parts may break the concrete`, `a marsh`.

Deterministic matching will resolve most of SciQ and comparatively little of OBQA. **An OBQA gold
such as `parts may break the concrete` is an annotator's phrasing of an idea, not a recallable
string** — a model can be entirely correct and share almost no tokens with it. This is the same
extractive-vs-deductive split documented in
[`docs/ex2_counterfactual/counterfactual_experiment_methodology.md`](../ex2_counterfactual/counterfactual_experiment_methodology.md) §5.1,
and it means **A′ on OBQA measures phrasing agreement as much as knowledge**. B′ is what makes
that interpretable, and the per-stage breakdown must be reported for OBQA or the number will be
misread.

### 4.4 Stage definitions

**Stage 1 — normalised exact match.** Case-fold; strip punctuation; collapse whitespace; drop
leading articles (`a`, `an`, `the`). Match iff the normalised strings are equal.

**Stage 2 — alias / morphological match.** Reuse `morphological_variants()` from
[`counterfactual_passage.py:56-84`](../../code/ex2_counterfactual/counterfactual_passage.py) — already
written, already used in a published experiment, deterministic and dependency-free. Accept if any
variant of the prediction equals any variant of the gold under stage-1 normalisation. Also accept
when the normalised prediction and gold differ only by a leading article or by singular/plural.
Reusing this function rather than writing a new normaliser keeps one matching semantics across the
codebase.

**Stage 3 — LLM-judge adjudication of the residual only.** Judge: `Qwen3-4B-Instruct-2507`, the
same local 4-bit checkpoint, so no new dependency or download. Judged in a fresh context, one item
per call, with the *question* supplied so that synonymy can be assessed in context:

```
System: You are grading a short-answer science exam. Decide whether the student's
answer means the same thing as the reference answer, in the context of the
question. Ignore spelling, capitalisation, and phrasing. A more specific or more
general answer counts as correct only if it identifies the same thing. Reply with
exactly one word: CORRECT or INCORRECT.

User: Question: {question}
Reference answer: {gold}
Student answer: {prediction}
```

Scored with the **same letter-logit technique already used throughout the repo** — restrict the
final-position logits to the first tokens of `CORRECT` / `INCORRECT` and softmax over the two.
This makes the judge deterministic, gives a calibrated confidence per verdict, and eliminates
free-text parsing from the judge itself.

**Judge validation is part of the deliverable, not optional.** A stratified sample of 100 judged
items will be hand-labelled and the judge's agreement reported. A self-judging setup (the model
grading its own output) is a real bias risk; without an agreement figure the judged tier is not
citable. If agreement is below ~90%, report the deterministic-only number as the headline and the
judged number as an upper bound.

### 4.5 Reporting

Every headline gets three numbers so a reader can choose their own strictness:

- `acc_strict` — stage 1 only
- `acc_normalised` — stages 1 + 2 (**proposed headline**)
- `acc_judged` — stages 1 + 2 + 3 (upper bound), with judge agreement quoted

---

## 5. Models and datasets

**Primary:** `Qwen3-4B-Instruct-2507`, 4-bit NF4, the exact configuration of every existing
Qwen run, so A′ is directly comparable to the published A. Both full splits: SciQ test (884) and
OBQA test (500).

**Not included:** the `KDA_small` encoders cannot do this experiment at all — three are
`AutoModelForMultipleChoice` heads that require a candidate set, and `t5-small-ssm-nq`, while
generative, is a closed-book QA model whose output would not be comparable. Free-response is
**structurally unavailable** to the KDA_small ensemble. This is worth stating in the eventual
report: the original KDA formulation cannot ask this question of its own simulated students.

**Optional extension:** `Qwen2.5-7B-Instruct` is already downloaded under `models/`, so a
second-model check costs only runtime.

---

## 6. Interpretation — what each outcome would mean

Let `Δ = Acc_A − Acc_A′` (MCQ zero-context minus open-ended zero-context).

> **Revised after the pilot.** The table below originally read Δ directly as a
> recognition-vs-recall gap. **That is only valid where the B′ ceiling is near 1.0.** B′ supplies
> the answer in the prompt, so it bounds what A′ could possibly score; where B′ is low, Δ is
> measuring phrasing disagreement, not knowledge. Measured pilot ceilings: **SciQ B′ = 1.000**
> (Δ is interpretable), **OBQA B′ = 0.280–0.640** (Δ is *not* interpretable as knowledge). Every
> reading below is therefore conditioned on the dataset's own ceiling, and the runner now prints
> `A′ relative to its own B′ ceiling` for exactly this reason.

| Outcome | Valid only if | Reading | Consequence |
|---|---|---|---|
| **Δ small** (≲ 0.10) on SciQ | B′ ≈ 1.0 ✅ | The model genuinely *knows* these facts; options add little | Saturation is real knowledge; closes off "the options do the work" |
| **Δ large** (≳ 0.30) on SciQ | B′ ≈ 1.0 ✅ | Much of the 95.4% is recognition/elimination | Revises the motivating claim; option-stripping becomes a route to restoring KDA's dynamic range |
| **Δ large on OBQA** | B′ ≈ 1.0 ❌ (**0.28–0.64**) | **Uninterpretable as knowledge** — dominated by annotator-phrasing disagreement | Report `A′ / B′` only; never quote OBQA Δ against MCQ A |
| **B′ ≈ A′ on either dataset** | — | The fact does not help even open-ended | Reinforces that OBQA `fact1` is deductive and often non-informative |

**The ratio to report instead of Δ**, per dataset: `acc_band_low(A′) / acc_band_low(B′)`. Pilot
values: SciQ **0.680 / 1.000 = 0.680**; OBQA **0.080 / 0.280 = 0.286**.

**The link to the existing work is direct.** `docs/ex3_student_simulation/student_persona_simulation_report.md` concludes
that breaking saturation "will need a mechanism that actually removes knowledge — context
ablation, counterfactual passages (ex2), or a genuinely weaker student model — rather than one that
asks the model to pretend." **Option-stripping is a fourth candidate mechanism**, and unlike
persona conditioning it removes a real affordance rather than requesting a performance. A large Δ
would make it the most promising of the four, because it needs no second model and no data
generation.

**One interpretive guardrail.** A′ has no chance floor, where A has 0.25. Some of Δ is arithmetic,
not substantive. The honest comparison is against what a *reasonable* student would produce, so
the report must present Δ alongside the per-stage matching breakdown and never quote A′ against A
as though both were on the same scale.

---

## 7. Implementation and effort

**New file:** `code/ex4_free_response/run_free_response.py`, following the established layout
(`--limit` smoke flag, `resolve`/`ensure_parent` from `utils.paths`, structured JSON + `.log`).
Reuses `load_model()` from `kda_qwen_eval.py` and `morphological_variants()` from
`counterfactual_passage.py`.

**Outputs:**
```
results/ex4_free_response/results_free_response_sciq.json
results/ex4_free_response/results_free_response_obqa.json
results/ex4_free_response/free_response.log
docs/free_response_report.md
```

Per item, the record stores: raw generation, post-processed prediction, normalised forms, the
matching stage that fired, per-stage verdicts, and judge probability where applicable — so every
accuracy number is traceable to a decision on a single item.

**Effort estimate** (RTX 3050 Laptop, 4 GB — the hardware all existing timings come from):

| Step | Estimate |
|---|---|
| Implement runner + 3-stage matcher | ~2–3 h |
| Smoke test (`--limit 50`, both cells) | ~10 min |
| Full generation, 2 cells × 1384 items | **~60–90 min** (greedy, ≤24 new tokens/item) |
| Judge pass over the residual | ~10–20 min |
| Hand-validate 100 judged items | ~1 h |
| Write the report | ~1–2 h |
| **Total** | **~1 working day**, of which ~2 h is machine time |

The generation estimate is extrapolated from the ex3 run's measured 1.79 s/question for six
full-context forward passes; short-prompt greedy decoding of ≤24 tokens is of the same order.

**Risks:**

- *Verbose output despite instruction.* Mitigated by the stop-at-newline rule and the 24-token
  cap; the raw generation is retained so drift is visible rather than silent.
- *Self-judging bias.* Mitigated by the mandatory agreement check in §4.4 and by reporting the
  deterministic tier separately.
- *OBQA phrasing ceiling.* Not fully mitigable; handled by reporting B′ and the per-stage
  breakdown rather than a single number.


---

## 8. Pilot results (n = 25 per dataset, both cells) — and required changes

Run this round to sanity-check the cascade before committing to the full split, per the staging
instruction.

```bash
uv run --active python code/ex4_free_response/run_free_response.py --limit 25 --tag pilot \
    --log-file results/ex4_free_response/free_response_pilot.log
```

**Built:** [`code/ex4_free_response/run_free_response.py`](../../code/ex4_free_response/run_free_response.py)
(generation + judging) and [`code/ex4_free_response/matching.py`](../../code/ex4_free_response/matching.py)
(the three-stage cascade, unit-tested against hand-written cases).
**Outputs:** `results/ex4_free_response/results_free_response_{sciq,obqa}_pilot.json`,
`results/ex4_free_response/free_response_pilot.log`.
Runtime **0.8 min** for 100 generations + 74 judge calls (0.71–0.77 s/item).

### 8.1 Headline pilot numbers

| Dataset | Cell | n | `acc_strict` | `acc_normalised` | `acc_judged` | 95% CI (judged) | MCQ equivalent | Δ |
|---|---|---:|---:|---:|---:|---|---:|---:|
| SciQ | A′ (no options, no fact) | 25 | 0.200 | 0.240 | **0.720** | [0.524, 0.857] | 0.954 | **+0.234** |
| SciQ | B′ (no options, fact) | 25 | 0.720 | 0.720 | **1.000** | [0.867, 1.000] | 0.998 | −0.002 |
| OBQA | A′ | 25 | 0.000 | 0.000 | **0.280** | [0.143, 0.476] | 0.826 | **+0.546** |
| OBQA | B′ | 25 | 0.040 | 0.080 | **0.440** | [0.267, 0.629] | 0.920 | +0.480 |

These are n=25 with intervals 30+ points wide. **They are a harness check, not a result.**

### 8.2 What worked

**Format compliance is a solved problem.** The risk flagged in §7 (verbose output despite
instruction) did not materialise at all:

| Dataset | Cell | mean words | max words | multi-line generations |
|---|---|---:|---:|---:|
| SciQ | A′ / B′ | 1.5 / 1.6 | 3 / 3 | 0 / 0 |
| OBQA | A′ / B′ | 2.2 / 2.2 | 5 / 13 | 0 / 0 |

The 24-token cap and the "answer only" instruction are sufficient; no stopping criteria or
retry logic is needed.

**SciQ B′ = 1.000 judged is the harness's sanity check.** With the passage in the prompt, every
open-ended SciQ answer was accepted. If the pipeline were broken this cell would not be perfect.

### 8.3 What broke — the cascade's central premise is falsified

§4.2 justified the cascade on the grounds that "the deterministic stages resolve the large, easy
majority" and the judge handles "the genuinely ambiguous residual". **That is not what happens.**

| Dataset | Cell | resolved deterministically | sent to judge |
|---|---|---:|---:|
| SciQ | A′ | 6 / 25 (24%) | **19 (76%)** |
| SciQ | B′ | 18 / 25 (72%) | 7 (28%) |
| OBQA | A′ | 0 / 25 (0%) | **25 (100%)** |
| OBQA | B′ | 2 / 25 (8%) | 23 (92%) |

The reasoning behind the §4.3 word-count argument was sound but incomplete. It predicted that
short gold answers would be easy to match — true — but assumed the model would *produce the gold
string*. In open-ended recall it produces a **synonym**:

| Gold | Model's answer | Deterministically matchable? |
|---|---|---|
| `oxidants` | `oxidizing agents` | no |
| `backbone` | `Vertebrae` | no |
| `sperm and eggs` | `gametes` | no |
| `regular array` | `lattice structure` | no |
| `highly viscous` | `High viscosity` | no |

Every one is a correct answer. No amount of normalisation or morphological variation bridges them
— the gap is lexical-semantic, not morphological. **The judge is therefore the primary grading
mechanism, not a residual adjudicator**, which is precisely the design §4.2 rejected as "makes the
headline number a function of an unaudited second model".

### 8.4 The judge is not stable enough to carry that load

Semantic equivalence is symmetric, so swapping *reference* and *student* should not change a
verdict. All 74 judged pilot items were re-scored with the arguments swapped:

| Metric | Value |
|---|---:|
| Judged items re-scored | 74 |
| **Verdict flips under swap** | **18 (24.3%)** |
| Symmetric agreement | 75.7% |

Examples:

| Dataset | Gold | Prediction | Forward | Reversed |
|---|---|---|---|---|
| SciQ | `tree rings` | `fossil records` | INCORRECT | CORRECT |
| SciQ | `about 2 km` | `A few kilometers` | INCORRECT | CORRECT |
| SciQ | `electroluminescence` | `Gas discharge` | CORRECT | INCORRECT |
| OBQA | `carbon` | `carbon dioxide` | INCORRECT | CORRECT |
| OBQA | `liquid` | `water` | CORRECT | INCORRECT |

The judge is *confident* while being unstable — mean `|p − 0.5|` ≈ 0.47–0.50, with only 1–3
ambiguous items per cell. Confidence is not reliability here. A grader that decides 76–100% of
items and flips a quarter of its verdicts under a semantically null transformation **cannot carry
the headline number as currently specified**.

Part of the asymmetry is traceable to the prompt itself: *"A more specific or more general answer
counts as correct only if it identifies the same thing"* asks a directional question, and
`carbon` → `carbon dioxide` is genuinely a different judgement in each direction.

### 8.5 OBQA: the phrasing ceiling is real and now measured

§4.3 warned that OBQA A′ would partly measure phrasing agreement. **B′ quantifies it:** with the
fact supplied, open-ended OBQA still scores only **0.440**. Since B′ has strictly more information
than A′, that is an empirical ceiling on what A′ could reach.

So OBQA's Δ of +0.546 is **not** a knowledge gap and must not be reported as one. Against its own
ceiling, A′ = 0.280 recovers 64% of what is achievable (0.280 / 0.440), and the remaining 56 points
of Δ are dominated by the model not guessing the annotator's phrasing (`parts may break the
concrete` answered as `root pressure`; `human planet rotation` answered as `noon`). SciQ has no
such problem — its B′ ceiling is 1.000.

### 8.6 Required changes before the full run

1. **Grade in both directions and report a band, not a point.** Run the judge forward and swapped;
   accept only unanimous CORRECT. This converts the 24.3% instability from hidden error into an
   explicit uncertainty interval:
   `[acc_both_agree_correct  ,  acc_either_correct]`
   with `acc_normalised` retained as a deterministic floor. Cost: 2× judge calls, which is
   negligible (judging was under a third of pilot runtime).

2. **Promote human validation from a side check to the gate.** §4.4 proposed 100 hand-labelled
   items as a check on a mechanism expected to handle a minority of cases. It now handles nearly
   all of them, so: hand-label **150 judged items stratified by dataset, cell, and forward/reverse
   agreement**, report Cohen's κ against the judge, and **do not publish `acc_judged` at all if
   κ < 0.7** — report the deterministic floor and the band instead.

3. **Add a second, independent judge.** `Qwen2.5-7B-Instruct` is already in `models/`. Self-judging
   bias was a modest concern when the judge was an adjudicator of last resort; with it deciding
   90%+ of OBQA it is a first-order threat. Cross-model agreement costs one extra pass.

4. **Report OBQA A′ relative to its B′ ceiling**, never against MCQ A alone. Every OBQA table must
   carry B′ alongside A′.

5. **Revise the §6 interpretation table.** It assumed Δ could be read directly as a
   recognition-vs-recall gap. That holds only where the B′ ceiling is near 1.0 — true for SciQ
   (1.000), false for OBQA (0.440).

### 8.7 Revised effort

| Step | Original | Revised | Why |
|---|---|---|---|
| Full generation, 2 cells × 1384 | 60–90 min | ~35 min | measured 0.71–0.77 s/item, faster than estimated |
| Judging | 10–20 min | ~25 min | now bidirectional |
| Second-judge pass | — | ~25 min | new, item 3 above |
| Hand validation | 1 h (100 items) | ~2 h (150 items) | now the validity gate |
| Analysis + write-up | 1–2 h | 2 h | band reporting, per-dataset ceilings |
| **Total** | ~1 day | **~1 day**, ~1.5 h machine | the added machine time is cheap; the added *human* time is the real cost |

### 8.8 Bottom line

The harness is working and format compliance is solved. The experiment is worth running. But the
grading design must change first: **as specified, the headline number would have been ~85%
determined by a judge that flips a quarter of its verdicts when you swap its two arguments.** The
pilot cost 0.8 minutes of GPU time and caught that before the full run — which is what it was for.


---

## 9. Bidirectional pilot and validation tooling (this round)

### 9.1 What was implemented

| Change | Where | Status |
|---|---|---|
| Bidirectional grading (forward + swapped) | `run_free_response.py`, `judge` call site | ✅ |
| Band reporting instead of a point estimate | `matching.py::accuracy_tiers` | ✅ |
| OBQA A′ always reported against its own B′ ceiling | `run_free_response.py` summary | ✅ |
| §6 interpretation table revised | this document | ✅ |
| Validation sampling + annotation sheet | `code/ex4_free_response/build_validation_sample.py` | ✅ built, not run by a human yet |
| Cohen's κ + pre-registered gate | `code/ex4_free_response/compute_kappa.py` | ✅ built, self-tested |
| Second judge (Qwen2.5-7B) | [`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) entry 3 | ⬜ queued, cloud |

The band is defined in `accuracy_tiers`:

```
acc_strict       stage 1 only                                     hardest floor
acc_normalised   stages 1-2, deterministic only                   defensible floor
acc_band_low     deterministic + judge CORRECT in BOTH directions  conservative
acc_band_high    deterministic + judge CORRECT in EITHER direction permissive
```

The width of `[acc_band_low, acc_band_high]` **is** the judge's order-instability, made explicit
rather than hidden inside a point estimate.

### 9.2 Bidirectional pilot results (n = 25 per dataset)

```bash
uv run --active python code/ex4_free_response/run_free_response.py --limit 25 --tag pilot_bidir \
    --log-file results/ex4_free_response/free_response_pilot_bidir.log
```

| Dataset | Cell | n | `acc_strict` | `acc_normalised` | **BAND [low, high]** | width | → judge | flip rate |
|---|---|---:|---:|---:|---|---:|---:|---:|
| SciQ | A′ | 25 | 0.200 | 0.240 | **[0.680, 0.800]** | 0.120 | 19 | 0.158 |
| SciQ | B′ | 25 | 0.720 | 0.720 | **[1.000, 1.000]** | 0.000 | 7 | **0.000** |
| OBQA | A′ | 25 | 0.000 | 0.000 | **[0.080, 0.320]** | 0.240 | 25 | 0.240 |
| OBQA | B′ | 25 | 0.040 | 0.080 | **[0.280, 0.640]** | 0.360 | 23 | **0.391** |

**Bidirectional grading did not reduce the instability — it localised it.** The overall flip rate
is unchanged at 24.3% (18/74; same judge, same items), which was expected: running the judge twice
cannot make it more self-consistent. What it buys is that the instability is now *visible per
cell*, and the picture that emerges is sharply dataset-dependent:

- **SciQ B′ is perfectly stable** (0.000 flips, band width 0) — with the passage in the prompt the
  judging task is easy and unambiguous.
- **SciQ A′ is usable** — 15.8% flips, band width 0.120.
- **OBQA is not usable as it stands** — B′ flips on 39.1% of judged items and its band spans
  0.280–0.640. A 36-point-wide interval on the *ceiling* cannot support any claim about A′.

This is a more actionable result than the single 24.3% figure from §8.4: the problem is
concentrated where gold answers are long annotator phrasings, exactly as §4.3 predicted.

### 9.3 The validation gate — built, ready to run

**Step 1 — generate the sheet** (already done for the pilot; re-run after the full run):

```bash
uv run --active python code/ex4_free_response/build_validation_sample.py --tag pilot_bidir --n 150
```

Stratifies over dataset × cell × (judge agreed / judge flipped), **taking every flipped item
first** since those carry the most information about reliability, then filling proportionally.
Seeded (`--seed`, default 20260904) and the seed is printed. Output:
`results/ex4_free_response/validation_sheet_<tag>.csv`.

For the pilot this yields all 74 judged items:

| Stratum | n |
|---|---:|
| sciq/A_prime/agree | 16 |
| sciq/A_prime/flip | 3 |
| sciq/B_prime/agree | 7 |
| obqa/A_prime/agree | 19 |
| obqa/A_prime/flip | 6 |
| obqa/B_prime/agree | 14 |
| obqa/B_prime/flip | 9 |

**Step 2 — label it.** Open the CSV and fill **only** the `human_verdict` column with `CORRECT` or
`INCORRECT`. The file carries its own instructions in a `#` header block, including the request
not to read the `judge_forward` / `judge_reverse` columns before deciding (hide them if your
editor allows). Rows are in randomised order to avoid ordering effects.

**Step 3 — score it:**

```bash
uv run --active python code/ex4_free_response/compute_kappa.py \
    --sheet results/ex4_free_response/validation_sheet_pilot_bidir.csv \
    --out results/ex4_free_response/kappa_report.json
```

It reports Cohen's κ against **two** candidate headline mechanisms — the forward-only verdict and
the both-directions-agree verdict — plus κ within each stratum, and then applies the gate:

| Result | Consequence, applied automatically |
|---|---|
| **κ ≥ 0.70** | Publish `acc_band_low` / `acc_band_high` as the headline, quoting κ |
| **κ < 0.70** | Publish `acc_normalised` (deterministic floor) as the headline; the band becomes a diagnostic only, and `acc_judged` is not quoted at all |

The threshold is **pre-registered in code** (`KAPPA_GATE = 0.70` in `compute_kappa.py`) so it
cannot be adjusted after the labels exist.

The scorer was verified end-to-end against a synthetic sheet simulating an annotator who agrees
with the judge 85% of the time: it returned κ = 0.645 and correctly reported GATE NOT PASSED.
**That was a self-test of the tooling, not a result** — no human labels exist yet.

### 9.4 What is still blocked, and on whom

| Blocker | Owner | Cleared by |
|---|---|---|
| Human validation gate (150 items, κ ≥ 0.70) | user | filling `validation_sheet_*.csv` and running `compute_kappa.py` |
| Qwen2.5-7B cross-judge comparison | user (Colab/Kaggle) | [`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) entry 3 — ~2 min of GPU time |

The full-scale run stays unlaunched until both clear. Machine time for the full run is ~1.5 h; the
reason to wait is not cost but that a headline number produced now would rest on a grader whose
reliability is unmeasured, and OBQA's band would be 36 points wide.

---

## 10. Qwen2.5-7B — cross-judge gate (Entry 3) and full-scale self-judged run (added 2026-09-24)

Two independent pieces of Qwen2.5-7B evidence now exist for this experiment. Both are reported
here without altering §1–§9's design, pilot results, or open items above.

### 10.1 Entry 3 — the cross-judge gate, run

[`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) §4 (Entry 3) has been run:
`results/ex4_free_response/judge_qwen2.5_7b_free_response_pilot.json` has Qwen2.5-7B judging, in
both directions, the same 74 residual (judge-tier) items from the Qwen3-4B bidirectional pilot
(§9.2) that Qwen3-4B judged. Per §4's own specified analysis, this reuses
[`compute_kappa.py`](../../code/ex4_free_response/compute_kappa.py) — writing Qwen2.5-7B's forward
verdict into the `human_verdict` column and Qwen3-4B's forward/reverse verdicts into
`judge_forward`/`judge_reverse` of a constructed sheet — so the same pre-registered `KAPPA_GATE =
0.70` machinery scores inter-judge agreement exactly as it would score human-vs-judge agreement.

**Inter-judge agreement, forward verdicts** (n = 74):

| | Qwen3-4B CORRECT | Qwen3-4B INCORRECT |
|---|---:|---:|
| **Qwen2.5-7B CORRECT** | 28 | 7 |
| **Qwen2.5-7B INCORRECT** | 10 | 29 |

| Metric | Forward-only | Both-directions-agree |
|---|---:|---:|
| Raw agreement | 77.0% | 77.0% |
| Cohen's κ | 0.541 | 0.545 |
| **`best_kappa`** | | **0.545** |

**The pre-registered gate is not cleared:** `best_kappa = 0.545 < KAPPA_GATE = 0.70` →
**GATE NOT PASSED.** Applying `compute_kappa.py`'s own decision rule (§9.3 / §4.3 above) to this
cross-model comparison the same way it applies to a human-vs-judge comparison: `acc_normalised`
(the deterministic floor) is what the gate rule would license as headline-worthy, not
`acc_band_low`/`acc_band_high`, for a run graded solely by either model against the other.

By stratum (forward-only κ): `obqa/A_prime` 0.335 (n=25), `obqa/B_prime` 0.569 (n=23),
`sciq/A_prime` 0.548 (n=19), `sciq/B_prime` 0.000 (n=7, all-INCORRECT-vs-mostly-CORRECT — a
degenerate small-n stratum). No stratum reaches 0.70 either.

**Per-model order-instability, for comparison** (flip rate under forward/reverse swap, same 74
items each model judged):

| | SciQ | OBQA | Overall |
|---|---:|---:|---:|
| Qwen3-4B (original judge, §8.4/§9.2) | 15.8% | 24.0–39.1%* | 24.3% |
| Qwen2.5-7B (this entry) | 11.5% | 25.0% | 20.3% |

\* Qwen3-4B's OBQA flip rate splits by cell in §9.2 (A′ 24.0%, B′ 39.1%); the 74-item pilot pool
mixes both. Qwen2.5-7B is somewhat more self-consistent overall (20.3% vs 24.3%) but is not
dramatically more stable, and the two models agree with each other (κ=0.545) distinctly less than
either model agrees with itself under a swap (i.e. both models are individually more self-
consistent than they are mutually consistent) — the instability §8.4 found is not simply "fixed" by
switching judges.

### 10.2 Full-scale Qwen2.5-7B generation + self-judge run

`RUN_QWEN2.5.md` Entry 4 was also run, but at **full scale** (884 + 500 items, not the 25-item
pilot Entry 4 itself specifies) and **self-judged** by Qwen2.5-7B — i.e. this is a model-swapped
repeat of the full free-response experiment §9 was built for, using Qwen2.5-7B as both generator
and judge, mirroring the original Qwen3-4B self-judged design. It is **not** the cross-judge check
in §10.1, and it does not depend on §10.1's gate outcome: it is graded by its own generator, exactly
as the Qwen3-4B pilot in §9.2 was.

```bash
# already run; see results/ex4_free_response/results_free_response_{sciq,obqa}_qwen2.5_7b.json
```

| Dataset | Cell | n | `acc_strict` | `acc_normalised` | **BAND [low, high]** | width | → judge | flip rate |
|---|---|---:|---:|---:|---|---:|---:|---:|
| SciQ | A′ | 884 | 0.512 | 0.535 | **[0.722, 0.804]** | 0.082 | 411 | 0.178 |
| SciQ | B′ | 884 | 0.805 | 0.825 | **[0.965, 0.984]** | 0.019 | 155 | 0.110 |
| OBQA | A′ | 500 | 0.036 | 0.036 | **[0.246, 0.440]** | 0.194 | 482 | 0.201 |
| OBQA | B′ | 500 | 0.108 | 0.110 | **[0.448, 0.660]** | 0.212 | 445 | 0.238 |

Against §9.2's Qwen3-4B **pilot** bands (n=25 each, reproduced here for reference):

| Dataset | Cell | Qwen3-4B pilot BAND (n=25) | Qwen2.5-7B full-scale BAND (n=884/500) |
|---|---|---|---|
| SciQ | A′ | [0.680, 0.800] | [0.722, 0.804] |
| SciQ | B′ | [1.000, 1.000] | [0.965, 0.984] |
| OBQA | A′ | [0.080, 0.320] | [0.246, 0.440] |
| OBQA | B′ | [0.280, 0.640] | [0.448, 0.660] |

Qwen2.5-7B's full-scale bands land close to, and on OBQA somewhat above, the Qwen3-4B pilot's
25-item bands — consistent given the pilot's wide intervals, but not a like-for-like comparison
(different n, different model as both generator and self-judge). **A′-relative-to-B′-ceiling**
(§6/§8.5 convention, `acc_band_low(A′) / acc_band_low(B′)`): SciQ **0.722 / 0.965 = 0.748**, OBQA
**0.246 / 0.448 = 0.549**. SciQ's ceiling is again near 1.0 (0.965), so its A′/ceiling ratio is
interpretable roughly as knowledge; OBQA's ceiling (0.448) is again well short of 1.0, so per §6's
own interpretive guardrail, OBQA's Δ against MCQ accuracy should not be read as a knowledge gap here
either — the same phrasing-ceiling problem §8.5 documented for Qwen3-4B persists for Qwen2.5-7B at
full scale.

**Runtime / memory:** SciQ 980.68 s (16.3 min, 1.048 s/item), OBQA 1703.99 s (28.4 min, 1.447
s/item); peak reserved VRAM 5.822 GB on the Kaggle 2×T4 setup for both runs.

### 10.3 Status note

Per the project's own pre-registered decision rule (§9.3, `KAPPA_GATE = 0.70` in
`compute_kappa.py`), the cross-judge check in §10.1 does not clear the gate (best κ = 0.545). The
human validation gate (§4.0 of `docs/NEXT_PHASE_HANDOFF.md`) separately passed at κ = 0.754 on the
both-directions-agree verdict, judging Qwen3-4B against a human. These are two different
comparisons — human-vs-Qwen3-4B-judge, and Qwen2.5-7B-vs-Qwen3-4B-judge — and the gate rule applies
independently to each; one passing does not imply the other would. Reproduction/artefacts for this
section: `results/ex4_free_response/judge_qwen2.5_7b_free_response_pilot.{json,log}`,
`results/ex4_free_response/results_free_response_{sciq,obqa}_qwen2.5_7b.json`,
`results/ex4_free_response/free_response_qwen2.5_7b.log`.
