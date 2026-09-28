# RQ1 Explained: Can LLM-Based Evaluators Tell Whether a Question Needs Its Material?

> **RQ1.** Do modern LLM-based evaluators reliably measure whether a multiple-choice question
> genuinely requires the target learning material?

**Short answer: no.** The rest of this report explains why, what the counterfactual experiments
added, and how far the conclusion can be trusted. Readers who want the full set of numbers should
see [`rq1_final_report.md`](rq1_final_report.md). Corrections to earlier drafts are listed in
[`rq1_final_report_corrections.md`](rq1_final_report_corrections.md).

## What was measured

The evaluator under study is **KDA** (Knowledge Dependent Answerability; Moon et al., EMNLP 2022). Its
idea is simple: a question genuinely depends on its learning material if students who cannot
answer it *without* the material can answer it *with* the material. KDA estimates this by
simulating students with language models. Each model answers every question twice:
- once with the question and options only;
- once with the target passage added.

KDA comes in several forms:
- **$KDA_{disc}$** uses right/wrong answers.
- **$KDA_{cont}$** uses the model's probability of the correct answer, and weights each model by
  how *unlikely* it was to answer correctly without the material. This is the "ignorance weight".
- **Adjusted variants** use the counterfactual test described in Part 2.

We applied KDA to two science benchmarks: SciQ (884 test questions) and OpenBookQA (OBQA, 500 test
questions). Three kinds of simulated student were used:
- the original paper's ensemble of four small encoder models (`KDA_small`);
- Qwen3-4B-Instruct;
- Qwen2.5-7B-Instruct.

No real students were tested. Every KDA value here is a model-based estimate.

---

## 1. Why the answer is "No"

### Reason 1: Modern LLMs already know most answers, so the metric has almost nothing left to measure

KDA can only learn something from questions a student gets **wrong without the material**. A
question the model already answers correctly contributes nothing: it drops out of both the
numerator and the denominator. Modern LLMs answer most of these questions without any passage.

| Simulated student | SciQ answered correctly without the passage | Questions left for $KDA_{disc}$ (SciQ) | OBQA answered without passage | Questions left (OBQA) |
|---|---:|---:|---:|---:|
| `KDA_small` ensemble | 45.48% | — | 34.80% | — |
| Qwen3-4B | 95.36% | 41 of 884 (4.6%) | 82.60% | 87 of 500 (17.4%) |
| Qwen2.5-7B | 95.02% | 44 of 884 (5.0%) | 85.20% | 74 of 500 (14.8%) |

With Qwen3-4B, the SciQ score of $KDA_{disc}$ = 0.9512 rests on 41 questions out of 884. The
number looks confident, and nothing in it reveals that 95% of the benchmark was silently ignored.
The same pattern appears with a second, larger model, so it is not specific to one solver.

Supplying the passage barely changes these models' answers. It raises Qwen3-4B's SciQ accuracy by
4.41 percentage points, against 43.10 points for the small encoder ensemble. The continuous version
does not rescue the situation, because the model's confidence is almost all-or-nothing: 87.9% of
SciQ questions receive a probability of exactly 1.0 without the passage. Sources:
[Qwen3-4B report](../ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md) §4–7,
[Qwen2.5-7B report](../ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md) §4–8.

We tried to recover a weaker student by prompting the model to role-play a beginner. It did not
work:
- Answering as separate personas, accuracy across beginner, intermediate and advanced differed by
  only 0.036 on SciQ.
- Answering all three personas in one prompt produced large gaps. These were an artifact of the
  order in which the personas answered: reversing the order moved the SciQ gap from +0.342 to
  +0.639 and flipped OBQA's from −0.228 to +0.408
  ([persona report](../ex3_student_simulation/student_persona_simulation_report.md) §3, §7).

### Reason 2: The score does not separate questions that need the material from questions that don't

This test uses the full SciQ and OBQA populations and the original four-model ensemble, with no
manual labels involved. A good measure of "needs the material" should be **lower** for questions
that are easy without the material. $KDA_{cont}$ shows no such pattern:

| Correlation of $KDA_{cont}$ with… | SciQ | OBQA |
|---|---:|---:|
| probability of answering correctly *without* the material | +0.075 | +0.301 |
| probability of answering correctly *with* the material | +0.863 | +0.878 |

The first correlation should be clearly negative. Instead it is near zero or positive. The score
mostly measures how easy a question is *once the passage is given*, which is a different property.

The same point shows up directly. Among the 10% of questions KDA rates as *most* dependent on the
material, the primary model still answers 41.6% (SciQ) and 42.0% (OBQA) correctly with no passage
at all. Questions the model answers correctly both with and without the passage score slightly
*above* the average (0.495 vs 0.471 on SciQ), though they plainly did not need it. Source:
[failure analysis](../ex5_failure_audit/rq1_test_split_failure_analysis.md) §4.

### Reason 3: The score reflects the simulated student more than the question

If KDA measured a property of the question, different simulated students should broadly agree.
They do not. On the same SciQ questions, individual students give KDA scores ranging from 0.2737
to 0.8146 ([reproduction summary](../ex1_reproduce_KDA/kda_reproduction_summary.md) §2.2).

Part of the reason is in the formula. The ignorance weight is supposed to emphasise questions a
student finds hard without the material. For two of the four small students, the weight barely
changes from question to question (standard deviation about 0.04–0.10), so it cannot tell
questions apart. For Qwen3-4B the weight averages 0.0458: close to zero almost everywhere. In
addition, about 30% of the $KDA_{cont}$ numerator (29.8% SciQ, 30.8% OBQA) comes from students who
were already correct without the material. Sources:
[failure analysis](../ex5_failure_audit/rq1_test_split_failure_analysis.md) Tables 8, 11;
[E1](../ex2_counterfactual/e1_counterfactual_llm_scale.md) §4.

### Reason 4: KDA does not agree with human judgement

The project's independent human check covered 100 SciQ questions. Two annotators rated each one
without seeing any model output or score. The question they answered was:

> **Imagine a typical secondary-school student who has studied general science but has NOT studied
> this particular topic, and who has NOT read the passage.** They see only the question and the four
> options. **How likely are they to pick the correct answer?**

The key measure is AUC: the probability that the metric ranks a randomly chosen human-labelled
"dependent" question above a "not dependent" one. 0.5 means chance.

| $KDA_{cont}$ vs human labels | Annotator 1 | Annotator 2 |
|---|---|---|
| AUC (95% CI) | 0.580 [0.459, 0.701] | 0.569 [0.420, 0.718] |
| AUC weighted to the full population | 0.455 | 0.387 |

Both confidence intervals include 0.5, and the population-weighted values fall below it. The
pre-registered primary test returned INCONCLUSIVE (p = 0.911 and 1.000). That test had 94% power
to detect a large difference, so this is weak evidence that no large difference exists, not just
a small sample. Source: [annotation results](../ex8_independent_labels/rq2_annotation_results.md)
§B1–B2, §D3.

### Reason 5: The original paper's own validation was weakest on exactly these benchmarks

The KDA paper reported a strong correlation with human-measured KDA overall (0.74 for
$KDA_{cont}$, 0.80 for $KDA_{disc}$). That figure is pooled across three datasets. On SciQ alone,
the correlations were 0.17 and 0.05, and neither is statistically significant. For human-written
questions like those in SciQ and OBQA, the correlation between $KDA_{cont}$ and expert ratings was
−0.01 overall. The authors also named both failure mechanisms in their own limitations:
- questions that can be answered from the question stem or by eliminating options;
- models that already know the fact from pretraining.

Source: [KDA paper notes](../papers/KDA_Paper_Documentation.md) Tables 4, 10; §4.

### How these reasons fit together (interpretation)

KDA measures a conditional quantity: how often the material helps *given that the student did not
already know the answer*. As models get stronger, that condition almost never holds (Reason 1).
The formula then falls back on how easy the question is with the passage (Reason 2), weighted by
factors specific to each simulated student (Reason 3).

The failure is therefore **insensitivity, not inversion**. KDA does not systematically score
material-independent questions *lower*. It simply cannot see them. An inverted metric could be
recalibrated; an insensitive one carries no signal to recover. Human agreement (Reason 4) and the
original paper's SciQ results (Reason 5) suggest this is not a new defect introduced by LLMs. It is
a validity gap the metric already had, which modern models make much larger.

---

## 2. What the counterfactual experiments showed

### The approach

Counterfactual testing asks directly whether a correct answer came from the passage. There are
three settings:
- **Setting A:** no passage.
- **Setting B:** the original passage.
- **Setting C:** a *counterfactual* passage. A simple string substitution rewrites the correct
  answer inside the passage as one of the question's wrong options. For example, "Dissociation is
  the separation of ions…" becomes "Combustion is the separation of ions…". No language model
  generates the text.

If a model follows the rewritten passage, the item is labelled **context-dependent**. If it still
gives the original answer, it is labelled **prior-dependent**: the model is answering from what it
already knows ([methodology](../ex2_counterfactual/counterfactual_experiment_methodology.md) §0–1).

This was used for three things:
1. to estimate how many "correct with the material" answers were really driven by the material;
2. to check whether KDA scores separate prior-dependent from context-dependent questions;
3. to build adjusted KDA estimators that discount or exclude prior-dependent questions.

### Main findings

**The manipulation works on SciQ.** When the passage is rewritten, the primary encoder model's
accuracy on the original answer falls to 0.083, and it picks the new counterfactual answer 0.853 of
the time ([OBQA/SciQ analysis](../ex2_counterfactual/counterfactual_obqa_analysis.md) §2.1). So on
SciQ the small models do read the passage.

**Many "correct with material" answers do not depend on the material.** Consider SciQ questions
answered correctly both with and without the passage:
- For the encoder ensemble, 22.3% are prior-dependent (21.5% after the leakage correction below).
- For Qwen3-4B, the share is 62.5%. On the eligible questions its accuracy with the passage is
  0.9977, but only 0.3988 once answers that ignore the passage are removed.

One concern is that a strong model might be correctly *rejecting* a false passage rather than
ignoring it. A follow-up test told the model explicitly to treat the passage as authoritative. Even
then, 46.0% remained prior-dependent. The true rate for Qwen3-4B therefore lies between 46.0% and
62.5%, at least double the encoder ensemble's rate. Sources:
[E1](../ex2_counterfactual/e1_counterfactual_llm_scale.md) §0,
[E2](../ex2_counterfactual/e2_prior_vs_rejection.md) §0.

**KDA does not rank prior-dependent questions lower.**
- SciQ questions labelled prior-dependent have a mean $KDA_{cont}$ of 0.474, essentially the same as
  the overall mean of 0.477.
- As a classifier between the two labels, $KDA_{cont}$ has an AUC of 0.5494 on SciQ, which is
  chance level ([failure analysis](../ex5_failure_audit/rq1_test_split_failure_analysis.md)
  Table 6; [P/S/F/D formulation](../ex6_psfd_score/psfd_formulation.md) §3).

**The adjusted estimators do not fix KDA.**
- The exclusion variant drops questions the ensemble labels prior-dependent (or unstable) and
  averages the rest. It returns 0.4793
  against an unadjusted 0.4767: almost no change, because the dropped questions scored the same as
  the kept ones.
- The hard and soft variants lower the overall level (0.3357 and 0.2641 on SciQ). Neither has been
  checked against human labels
  ([methodology](../ex2_counterfactual/counterfactual_experiment_methodology.md) §6;
  [P/S/F/D formulation](../ex6_psfd_score/psfd_formulation.md) §1.2).

**The label depends on the solver, not only on the question.** The encoder ensemble and Qwen3-4B
saw identical counterfactual passages for 860 SciQ questions, yet they agree on the label at
Cohen's κ = 0.046, barely above chance. Of the 669 questions the ensemble called context-dependent,
Qwen3-4B called 58.6% prior-dependent ([E1](../ex2_counterfactual/e1_counterfactual_llm_scale.md) §3).

### Known problems with counterfactual passage construction

The substitution is purely lexical. It has no grammar check, and no check that the rewritten
passage still supports exactly one option. Three problems follow.

1. **Some rewritten passages still state the original answer.** The original leakage check looked
   only for the exact answer string. That string is precisely what the substitution had just
   replaced, so the check reported 0 of 860 by construction. A later rescan also looked for
   inflected forms, fused words and partially rewritten clauses. It found:
   - 30 SciQ passages (3.5%) that still state the correct answer where the question is answered;
   - 26 of those 30 in the "exact" tier, which had been treated as the cleanest.

   These leaks bias the label. The ensemble labels a leaky item prior-dependent 46.7% of the time,
   against 11.6% for clean items, a factor of 4.0. Removing the 30 items lowers the ensemble's
   overall prior-dependent share only from 13.37% to 12.17%. The source concludes that no
   published conclusion changes. The 30 is a lower bound from a lexical check, and paraphrased
   leaks would be missed ([leakage rescan](../ex7_counterfactual_construction/residual_leakage_rescan.md)
   §3, §5–6).
2. **Human reading confirms the problem in a sample.** In a 55-item human check, 17 of 57
   prior-dependent labels (29.8%) came from perturbations that had failed, rather than from a
   solver overriding the passage
   ([annotation results](../ex8_independent_labels/rq2_annotation_results.md) §B4b). This figure
   applies to that sample, not to the whole corpus.
3. **The method does not suit OpenBookQA.** OBQA facts are general rules ("predators eat prey"),
   not passages that contain the answer word. Only 145 of 500 OBQA questions could be rewritten at
   all. Of the context-dependent labels, 70.8% rest on passages that are not well-formed English
   (3.6% on SciQ). OBQA's counterfactual track was closed as a stress test; its figures are
   treated as caveats rather than conclusions
   ([methodology](../ex2_counterfactual/counterfactual_experiment_methodology.md) §0;
   [construction audit](../ex7_counterfactual_construction/counterfactual_construction_audit.md) §11).

### What can and cannot be concluded

**Can be concluded, for SciQ:**
- The counterfactual probe works: models do follow the rewritten passage.
- A substantial share of with-material successes are driven by prior knowledge, and the share is
  much larger for the modern LLM.
- KDA does not assign lower scores to the questions the probe identifies. This point survives the
  construction problems: on the audit's leak-free subset the AUC stays at chance (0.5507)
  ([construction audit](../ex7_counterfactual_construction/counterfactual_construction_audit.md) §3.4).

**Cannot yet be concluded:**
- Precise prior-dependence rates for Qwen3-4B. The leakage rescan covered only the encoder models.
- Any reliable OBQA result.
- That the prior-dependent label is a property of the question. Two solvers disagree almost
  completely.
- That any adjusted KDA estimator is valid, since none has been checked against human judgement.

---

## 3. Main limitations

These limitations affect how strongly the RQ1 conclusion can be stated.

- **No real students.** KDA was designed around real student responses. Here it is always
  estimated from models. Comparison with human-measured KDA exists only in the original paper's
  data.
- **The human check is small.**
  - It covers 100 SciQ questions (plus 55 for the counterfactual reading), on one dataset only.
  - Annotator agreement was κ = 0.680 on the full sample, below the 0.70 target. It passed only on
    the pre-registered 30-item block (0.870), and the gate was recorded as passed.
  - The human question asks how answerable an item is *without* the passage. It does not test
    context-dependence directly
    ([annotation results](../ex8_independent_labels/rq2_annotation_results.md) §B5–B6).
- **"Known without the material" is measured in multiple-choice form, which overstates recall.**
  - Asked the same SciQ questions without options, Qwen2.5-7B's recall is estimated at 0.722–0.804,
    against 0.9502 with options.
  - That estimate relies on an LLM judge that did not pass the agreement check (κ = 0.545 against a
    second judge), so it is indicative only
    ([free-response experiment](../ex4_free_response/plan_option_free_response_experiment.md)
    §10).
  - The multiple-choice numbers include answers found by eliminating options, not only knowledge.
- **One configuration per modern model.** Each LLM was run once, with no correction for answer-order
  bias and no calibration check. The collapse in Reason 1 follows from the no-passage accuracy
  alone and would survive such changes, but the exact KDA values would move.
- **Part of the "flagged question" evidence is not independent of KDA.** Some Reason 2 comparisons
  use a pool of flagged questions, and two of the flagging criteria are defined by high KDA scores.
  The correlations and the top-decile result do not use this pool and are unaffected.
- **Scope.** Two English science benchmarks. All results are correlational.

Corpus quality was also checked, as a possible alternative explanation. Confirmed question–passage
mismatches affect at least 12 of 884 SciQ questions (1.36%), far too few to explain the results
([corpus-defect audit](../ex5_failure_audit/corpus_defect_audit.md) §8).

---

## 4. Conclusion

KDA-style evaluators do not reliably measure whether a multiple-choice question requires its
learning material. On SciQ and OpenBookQA, five lines of evidence point the same way:
- Modern LLMs already answer 95% of SciQ questions without the passage, leaving KDA with about 5%
  of the benchmark to score.
- The score tracks how easy a question is *with* the passage, not whether the passage is needed.
- Different simulated students give very different scores to the same questions.
- The score does not agree with independent human judgement.
- The original paper's own validation was already weak on SciQ.

The counterfactual experiments strengthen this picture while showing their own limits. They show,
on SciQ, that many correct answers with the passage come from prior knowledge. The share is at
least 46% for Qwen3-4B, and KDA gives those questions the same scores as the rest. But the
counterfactual passages are built by simple string substitution, and a small number still leak the
answer. The resulting labels depend on which model reads them, and the method does not work on
OpenBookQA. The counterfactual results therefore support the RQ1 answer but cannot yet provide a
validated replacement metric.

**Answer to RQ1: No.** Modern LLM-based evaluators of the KDA family do not reliably measure
whether a question genuinely requires the target material. The failure is insensitivity rather
than inversion. It grows as the simulated students become more capable, and it reflects a validity
gap already visible in the metric's original validation. The main open limitation is that the
human-grounded evidence comes from a single dataset and a small sample, and no real-student KDA
measurement exists.
