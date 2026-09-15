# Annotation Programme — Final Conclusion (Task A and Task B)

**Written:** 2026-09-11, after both annotation tasks completed.
**Sources:** rubric [`docs/guides/ANNOTATION_GUIDE.md`](../guides/ANNOTATION_GUIDE.md) · Task A result
[`NEXT_PHASE_HANDOFF.md`](../NEXT_PHASE_HANDOFF.md) §4.0 · Task B result
[`corpus_defect_audit.md`](corpus_defect_audit.md) §8.
**Artifacts every figure is read from:** `results/ex4_free_response/kappa_report.json`,
`results/ex5_failure_audit/corpus_defect_kappa.json`,
`results/ex5_failure_audit/corpus_defect_adjudication.json`.

> **Scope note.** This file lives under `docs/ex5_failure_audit/` because that is where it was
> requested, but Task A belongs to Experiment 4 and Task B to Experiment 5. It is a cross-package
> synthesis, not an ex5 document.
>
> **No annotation was performed and no result was changed to produce this file.** Every number below
> is read from a committed artifact. Throughout, claims are marked **[D]** demonstrated directly,
> **[I]** reasonably inferred, or **[—]** explicitly out of scope.

---

## Executive Summary

Two human annotation passes were run against two different objects. They are unrelated by design and
reach opposite verdicts.

| | **Task A** — judge validation | **Task B** — corpus defect |
|---|---|---|
| Object under test | the **LLM judge** (Qwen3-4B) in Experiment 4 | the **SciQ corpus** |
| n | 74 items | 33 items censused, 15 double-annotated |
| Result | κ = **0.754** (both-directions) / **0.614** (forward-only) | κ = **0.857**, agreement 0.933 |
| Gate | κ ≥ 0.70 pre-registered → **PASSED**, on one mechanism only | none by design |
| Verdict | **conditionally reliable** — trustworthy only where the judge is self-consistent | **the corpus is sound**: 12/884 = 1.36% defective, 98.64% clean |

**The headline is the asymmetry.** The dataset turned out to be in good shape, and its small defect
is fully enumerated and automatically detectable. The *evaluator* is the weaker component: it is
reliable on the 76% of items where it is self-consistent (κ = 0.816) and worthless on the 24% where
it flips under reference/student swap (κ = 0.069). The measurement instrument, not the data, is what
needs work.

Task B also produced the project's **first inter-annotator κ of any kind**, and in doing so overturned
one previously published label — lowering the SciQ mismatch rate from 13/884 (1.47%) to 12/884
(1.36%).

---

## Task A Conclusion

### What it directly demonstrated [D]

**The pre-registered gate passed, but on one mechanism only.** `KAPPA_GATE = 0.70`
(`compute_kappa.py:38`, marked *"do not change after seeing results"*) was cleared by
`best_kappa = 0.7535`, so `gate_passed = true`.

| Mechanism | n | Agreement | κ |
|---|---:|---:|---:|
| Forward-only verdict | 74 | 0.811 | **0.614** — *fails* |
| Both-directions-agree | 74 | 0.892 | **0.754** — passes |

The forward-only verdict is the one that would be quoted by default, and it fails. The scorer takes
the max of the two (`compute_kappa.py:151-157`), so the pass is real but narrow: **what it licenses
is specifically the both-directions band, not the forward verdict, and not `acc_judged`.**

**Reliability is conditional on the judge's self-consistency** — the substantive finding, and exactly
the case `compute_kappa.py`'s docstring (lines 13–16) anticipated in advance:

| Stratum | n | Agreement | κ |
|---|---:|---:|---:|
| `agree` — judge self-consistent | 56 | 0.911 | **0.816** |
| `flip` — judge unstable under swap | 18 | 0.500 | **0.069** |

κ = 0.069 is indistinguishable from chance. The judge is not uniformly mediocre; it is good in one
regime and useless in the other. That is why the both-directions mechanism works — discarding flipped
items is precisely what it does.

**The judge's errors are directional, not noise.** On the forward verdict, human `INCORRECT` / judge
`CORRECT` occurs **13 times against 1** in the reverse direction (confusion 22 / 1 / 13 / 38). **The
judge over-credits.** 14 of 74 items disagree on forward, 8 of 74 on both-directions (20 / 3 / 5 / 46).

**The human labelling was internally consistent.** Eight (question, gold, model) triples appear twice
in the sheet across different cells; all eight pairs were labelled identically — 100% intra-annotator
consistency. The disagreement with the judge is a genuine criterion difference, not annotator noise.

### What can reasonably be inferred [I]

- The 13:1 over-crediting asymmetry is the signature a **self-judging bias** would produce — the judge
  is Qwen3-4B grading Qwen3-4B's own output. It is equally consistent with a plain criterion
  difference (the human resolved general-vs-specific cases strictly, per guide §1.3(c)). **These two
  explanations are not separable by Task A**; that is what the Qwen2.5-7B cross-judge gate exists for.
- Because the judge decides 76–100% of items in the pilot, a conditional reliability result is close
  to a conditional result about the whole experiment: the deterministic matching tiers resolve too
  little to carry the headline alone.
- Two per-stratum κ values should not be read as signal: `sciq/A_prime/flip` (n=3, κ = −0.500) and
  `sciq/B_prime/agree` (n=7, κ = 0.000 at 85.7% agreement, a degenerate-margin artifact).

### What remains out of scope [—]

- **Self-judging bias is not ruled out.** Task A checks the judge against a human; it cannot check it
  against a different model. `RUN_QWEN2.5.md` Entry 3 is that check and is **still outstanding**.
- **Scale.** n = 74, not the intended 150, because only 74 pilot items had been judged
  (`build_validation_sample.py:148` defaults to `--n 150`). The flip stratum rests on 18 items.
- **Whether `acc_judged` is publishable.** It is not, on this evidence. The gate licenses the band.
- Task A says nothing about KDA saturation (RQ1), the counterfactual method (RQ2), or downstream
  quiz quality (RQ3).

---

## Task B Conclusion

### What it directly demonstrated [D]

**The SciQ corpus is sound, and its defect is smaller than previously published.**

| | |
|---|---:|
| Confirmed topic mismatches | **12 / 884 = 1.36%** |
| Clean passages | **872 / 884 = 98.64%** |
| Previously published (single annotator) | 13 / 884 = 1.47% — **superseded** |
| OpenBookQA | **0 / 500**, unchanged |

Confirmed defective item ids: `23, 153, 203, 306, 412, 598, 602, 610, 655, 739, 746, 754`.

**Inter-annotator agreement is high, as the protocol predicted it should be.** On the pre-registered
15-item subset: κ = **0.857**, observed agreement **0.933**, expected 0.533, confusion 9 / 1 / 0 / 5,
`unlabelled_by_B` empty. `corpus_defect_audit.md:278` argued that because *"is this passage about this
question?"* is an objective judgement, a low κ would indict the rubric rather than the sample. κ came
out high, so **the rubric holds.**

**The double annotation did real work: it overturned a published label.** One disagreement across the
15 (and one across all 21 shared items, agreement 0.952, descriptive only). Item **#584** — annotator 1
`MISMATCH`, annotator 2 `ON_TOPIC`, **adjudicated `ON_TOPIC`**. The passage is not confined to
respiratory anatomy; it also covers the mammalian circulatory system and states the keyed answer
outright. Annotator 1's recorded rationale characterised only the passage's opening two sentences.

**The overturned label was the LLM rater's.** Annotator 1 is an earlier LLM agent hand-check;
annotator 2 is the human. This is the error class the double annotation existed to catch, and it was
caught in the automated pass.

**A previously unexamined arm is clean.** The 12 items that entered via the Setting-C-ineligibility
screen alone (`33, 54, 183, 237, 381, 386, 594, 639, 760, 762, 843, 844`) had never been hand-checked
by anyone. The census found **zero** mismatches among them. Consequently all 12 confirmed mismatches
are Setting-C ineligible (100%, up from 92.3%).

### What can reasonably be inferred [I]

- **The overlap < 0.40 screen alone would have sufficed.** Every confirmed mismatch is caught by both
  screens, and the ineligibility-only arm contributed nothing. The union screen was the right
  conservative choice *ex ante*, but a future audit on another corpus can use the cheaper detector.
- **Corpus defect is not a limiting factor for this project.** At 1.36%, fully enumerated and
  automatically detectable, it cannot account for any headline effect. This closes the question rather
  than opening work.
- **The κ generalises to the rubric, not to the whole sheet.** 0.857 on an objective binary question
  supports the rubric's clarity; it does not independently validate the 12 ineligibility-only items,
  which have a single reading.

### What remains out of scope [—]

Per `corpus_defect_audit.md` §8.6, carried forward verbatim in substance:

- **One rater is an LLM agent.** This is human-vs-LLM κ, not human-vs-human.
- **n = 15.** One disagreement moves κ by roughly 0.06; never quote 0.857 without its n.
- **The rate is a lower bound.** Items above 0.40 overlap were sampled (0 of 20 mismatched), never
  censused. A large undercount is structurally unlikely but not excluded.
- **Semantic mismatch is invisible to this screen.** A passage that is on topic but does not support
  the keyed answer passes undetected. Nothing in Task B addresses that class.
- **Mis-key rate was deliberately dropped** (`corpus_defect_audit.md:264`) — not answerable at
  affordable n.
- **The two annotators saw different fields:** annotator 2's blind sheet omitted `gold_answer` by
  design, since the rubric does not use it.

---

## Combined Conclusion

### The two tasks separate data quality from measurement quality, and they come apart [D]

The dataset passed; the evaluator passed only conditionally. Stated plainly:

- **98.64% of SciQ passages are on topic**, and the 1.36% that are not are listed by id.
- **The LLM judge agrees with a human at κ = 0.816 where it is self-consistent and κ = 0.069 where it
  is not**, and it errs toward over-crediting by 13:1.

So when an Experiment-4 number is uncertain, the dominant term is the judge, not the corpus. [I]

### A concrete interaction, visible only because both tasks ran [D, n=1]

SciQ item **23** appears in both tasks — it is one of the 12 confirmed defective items (an
osteoporosis question paired with a passage about fungi), *and* it appears twice in Task A's sheet
(`row_id 13` in cell A′, `row_id 42` in cell B′).

In cell B′ the passage **is** supplied to the model. The model answered `fractures` against gold
`bone fractures`; both the human and the judge scored it `CORRECT` — correctly, under the Task A
rubric, since the answers mean the same thing. But the supplied passage is about fungi and contains
nothing about osteoporosis. **The credit was earned entirely from parametric prior; no reading of the
passage could have produced it.**

This is one item, so it is an illustration and not a rate. [—] But it shows that **Task A's rubric
will score such items CORRECT by construction**, which means B′ accuracy can bank credit that the
passage could not have conferred. At full scale that contamination is exactly the 12 enumerated items
of 884 (1.36%) — small, known, and trivially removable. [I]

It is also a miniature of the project's RQ1 thesis arriving from an unexpected direction: an item
where the material demonstrably cannot support the answer, answered correctly anyway.

### Both tasks found the same weakness in automated judgement, in different forms [I]

Task A's judge over-credited answer equivalence; Task B's LLM annotator called a mismatch on a passage
whose later sentences stated the answer. The *directions* differ, so this is not one bias. The common
factor is that **both LLM errors came from deciding on partial evidence** — and in both cases a human
pass caught it. That is an argument for keeping a human rater on the project's remaining single-pass
artifacts, independent of either κ value.

### What neither task establishes [—]

Neither task addresses: KDA saturation (RQ1), the counterfactual intervention (RQ2), downstream quiz
quality (RQ3), self-judging bias, OpenBookQA's corpus quality beyond the existing 0/500 sample, or
any accuracy figure at full scale. Both are validation passes on instruments and data, not results
about the research questions.

---

## Key Takeaways / Implications for Next Phase

**1. The full-scale free-response run is still blocked — Task A clearing does not release it. [D]**
`RUN_QWEN2.5.md:24` states the run is blocked on *"(a) the human validation gate … **and** (b) entry
3"*, and `plan_option_free_response_experiment.md:13` says *"Do not launch the full run until both
clear."* Gate (a) is cleared. **Entry 3 — Qwen2.5-7B as an independent second judge, ~2 min of cloud
GPU — remains outstanding and is the only item on the critical path.** Task A's 13:1 over-crediting
asymmetry strengthens rather than weakens the case for it, since that is the signature self-judging
bias would produce.

**2. Publish the both-directions band. Not the forward verdict, not `acc_judged`. [D]**
Forward-only κ = 0.614 fails the gate. Any Experiment-4 headline must quote κ = 0.754 alongside the
band, and should state that the forward mechanism did not clear.

**3. Report the judge's reliability as conditional, and prefer the both-directions mechanism. [I]**
κ = 0.816 / 0.069 across the agree / flip split is a usable result and the cheapest available
mitigation: the both-directions verdict discards the regime where the judge is worthless. The
24.3% flip rate is the defect to fix if the judge is to be improved rather than gated.

**4. Exclude or flag the 12 defective SciQ items in any passage-dependent analysis. [I]**
They are enumerated (`23, 153, 203, 306, 412, 598, 602, 610, 655, 739, 746, 754`). Item 23 shows the
mechanism concretely. This affects B′-style cells and any Setting-B/C computation that assumes the
passage supports the answer. Cost: a 12-id filter.

**5. Treat corpus defect as closed for SciQ; do not re-audit. [I]**
1.36%, fully enumerated, automatically detectable, and the cheaper overlap < 0.40 screen alone would
have found all of it. `corpus_defect_audit.md` §6 step 4 already required nothing for OpenBookQA.

**6. Re-validate at n = 150 after the full run, and over-sample the flip stratum. [D]**
`NEXT_PHASE_HANDOFF.md` §4.5 step 5: re-run `build_validation_sample.py --tag full --n 150` once a
full-scale judged pool exists, which is when stratified sampling finally does real work. The specific
weakness to target is the flip stratum at n = 18.

**7. The project's "no κ anywhere" caveat is now partially lifted — but only partially. [D]**
Task B is the first inter-annotator κ in the project. It remains human-vs-LLM at n = 15.
`passage_overlap_handcheck.json`'s own `_provenance` still says "no kappa" and is now out of date;
`corpus_defect_audit.md` §8 is the authority. Every *other* hand-annotated artifact in the repo
(`both_wrong_followup.json`, `obqa_new_admission_labels.json`) remains single-pass with no second
rater — and Task B's overturned label is direct evidence that such passes do contain correctable
errors.

**8. The blind-annotation workflow is now reusable tooling. [D]**
`build_blind_sheet.py` + `merge_blind_verdicts.py` (Task A, `pair_key` join) and
`build_blind_corpus_sheet.py` + `kappa_corpus_defect.py` (Task B, `item_id` join) implement
column-stripping, lock-before-compare, assertion-checked merges, and overwrite refusal. Any future
annotation should reuse them rather than hand-building a sheet — which is the lesson
`provenance_rq1_flagged_questions.md` already paid for once.
