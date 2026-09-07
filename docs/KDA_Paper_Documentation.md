# Evaluating the Knowledge Dependency of Questions — Paper Documentation

---

## 1. Executive Summary & Core Content

**Title:** Evaluating the Knowledge Dependency of Questions

**Authors & Affiliation:**
- Hyeongdon Moon*† (Riiid AI Research)
- Yoonseok Yang*† (UC Berkeley)
- Jamin Shin† (currently NAVER AI Lab; work done at Riiid AI Research)
- Hangyeol Yu (Riiid AI Research)
- Seunghyun Lee (Riiid AI Research)
- Myeongho Jeong (Riiid AI Research)
- Juneyoung Park (Riiid AI Research)
- Minsam Kim† (Riiid AI Research)
- Seungtaek Choi† (Riiid AI Research)

*Equal contribution. †Corresponding authors.
Code released at: https://github.com/riiid/question-score

**Venue / Year:** Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP 2022), pages 10512–10526, December 7–11, 2022.

**Research Question(s) & Problem Formulation:**
Automatic Question Generation (AQG), specifically Multiple-Choice Question (MCQ) generation, has been evaluated almost exclusively with n-gram based similarity metrics (BLEU, ROUGE, METEOR, BERTScore) that compare generated questions/distractors to a "gold" reference. The paper identifies two core problems with this approach:
1. Reliability depends entirely on the quality of the reference dataset.
2. N-gram similarity does not measure whether a question actually functions as a valid **assessment tool** — i.e., whether a student who knows the target fact can answer it, and a student who doesn't know it, cannot.

A prior "Answerability" metric (Nema and Khapra, 2018) attempted to address this but (a) did not tie answerability specifically to knowledge of the target fact, and (b) required human-authored references. The paper's core research questions are:
- **RQ1:** Can automated metrics (KDA_disc, KDA_cont) replicate human-measured KDA when PLMs are substituted for students?
- **RQ2:** Do KDA, KDA_disc, and KDA_cont correlate meaningfully with real-world educators' judgments of MCQ usability?

**Proposed Methodology:**
The paper introduces **Knowledge Dependent Answerability (KDA)** — a reference-free metric measuring the probability that a student who *doesn't* know the target fact will answer incorrectly, but *will* answer correctly once given the fact. This is estimated via human survey responses (before/after fact exposure) using Equation 1. Because human trials don't scale, the authors propose two **automatic, PLM-based surrogates**:
- **KDA_disc** (discrete): Replaces human binary correctness with PLM binary correctness (whether the correct answer has the max logit), with and without the fact prompted.
- **KDA_cont** (continuous): Uses PLM softmax probabilities directly (rather than binarized argmax), producing a smoother, more stable estimate — especially valuable because KDA_disc's denominator can become zero if all models answer correctly without the fact.

18 different pre-trained language models (T5 closed-book QA models, BERT, RoBERTa, MPNet, SciBERT, XLNet, BioBERT, DistilBERT/DistilRoBERTa, ALBERT, MatSciBERT) are used as "student proxies," queried both with and without the target fact prepended to the question.

**Main Contributions:**
- Proposes **KDA**, a novel, reference-free evaluation criterion for MCQ generation quality tied directly to whether a question requires knowledge of a specific target fact to answer.
- Proposes two **automatic approximations**, KDA_disc and KDA_cont, that replace costly human trials with PLM-based "solvers."
- Validates both metrics through **large-scale human studies** (116 student participants solving 480 MCQs) and **expert studies** (7 secondary-school science teachers rating 96 MCQs), showing strong correlation with both human-measured KDA and expert Likert usability scores — substantially outperforming BLEU/ROUGE/METEOR.
- Shows that combining KDA metrics with n-gram similarity metrics produces even stronger predictive power for expert-labeled MCQ quality measures (e.g., Accept, Irrelevancy).
- Releases open-source code and model weights for computing KDA_cont/KDA_disc.

---

## 2. Detailed Recreation of All Tables

### Table 1: Example questions from each dataset

| | OBQA | TabMCQ | SciQ |
|---|---|---|---|
| **Fact** | predators eat prey | Urban sprawl creates thermal pollution | Plant hormones are chemical signals that control different processes in plants. |
| **Question** | Predators eat | What type of pollution does Urban sprawl create? | What chemical signals in plants control different processes? |
| **Answer** | bunnies | thermal pollution | plant hormones |
| **Distractors** | lions, humans, grass | air pollution, radioactive pollution, noise pollution | produce hormones, nitrogen hormones, Human Hormones |

**Key Takeaway:** Illustrates the structural format (fact → question → answer → 3 distractors) shared across the three evaluation datasets used throughout the paper.

---

### Table 2: Baseline Models for QG/DG

| Generator | Distractors | Question Stem |
|---|---|---|
| Human | Human | Human |
| KDDG | KDDG | Human |
| T5DG | T5-DG | Human |
| QDG | T5-DG | T5-QG |

**Key Takeaway:** Defines the four MCQ generation pipelines compared in experiments — ranging from fully human-authored questions to fully model-generated (question stem + distractors) questions.

---

### Table 3: BLEU scores for T5-DG in RACE

| | BLEU1 | BLEU2 | BLEU3 | BLEU4 |
|---|---|---|---|---|
| D1 | 46.59 | 38.33 | 34.31 | 32.02 |
| D2 | 25.8 | 20.08 | 17.58 | 16.15 |
| D3 | 28.33 | 23.07 | 20.73 | 19.46 |

**Key Takeaway:** Shows the n-gram quality of the three generated distractors (D1, D2, D3) from the fine-tuned T5-Large distractor generator (T5DG) on the RACE test set; the first distractor (D1) scores notably higher than D2/D3.

---

### Table 4: Pearson Correlation between KDA and other automatic evaluation metrics

| | OBQA | TabMCQ | SciQ | All |
|---|---|---|---|---|
| **KDA_cont** | **0.73**\*\* | 0.16 | 0.17 | 0.74\*\* |
| **KDA_disc** | 0.71\*\* | **0.3**\*\* | 0.05 | **0.8**\*\* |
| BLEU | 0.29\*\* | 0.14 | 0.16 | 0.26\*\* |
| ROUGE-L | 0.29\*\* | 0.14 | **0.18**\* | 0.27\*\* |
| METEOR | 0.28\*\* | 0.12 | 0.14 | 0.21\*\* |

*Single (double) asterisk denotes p-value under 0.05 (0.01). Bold = best correlation per dataset column.*

**Key Takeaway:** KDA_cont and KDA_disc dramatically outperform n-gram metrics in correlating with human-measured KDA, especially in the aggregate ("All") column (0.74–0.80 vs. 0.21–0.27 for n-gram metrics). Correlation strength varies by dataset — strongest for OBQA, weakest for TabMCQ/SciQ (explained by those datasets' questions being "too easy").

---

### Table 5: Pearson Correlation of KDA_cont with KDA and expert Likert score by model size

| Size and number of LMs | KDA | Likert |
|---|---|---|
| LMs < 1GB (4 LMs) | 0.65 | 0.36 |
| LMs < 1.5GB (4 LMs) | 0.71 | 0.38 |
| LMs < 1.5GB (11 LMs) | 0.73 | 0.39 |
| All LMs (18 LMs) | 0.74 | 0.43 |

**Key Takeaway:** Both correlation with human KDA and with expert Likert scores improve monotonically as more/larger PLMs are used as "solver" proxies, suggesting better student-imitation with more capable and diverse language models.

---

### Table 6: Response types for the classroom-usability Likert question

*"On a scale of 1–4, how would you evaluate this question to be used in a classroom to test the fact below?"*

| Score | Description |
|---|---|
| 4 | **[Strongly Agree]** This question can be readily used in the classroom. |
| 3 | **[Agree]** Despite some minor flaws, I'm willing to use this question to test the fact in a classroom |
| 2 | **[Disagree]** This question has some major flaws that needs to be revised for educational use in a classroom. |
| 1 | **[Strongly Disagree]** This question should be changed completely to be used in a classroom. |

**Key Takeaway:** Defines the 4-point Likert scale used by expert teacher-annotators to rate MCQ classroom usability — the ground-truth signal used for RQ2 validation.

---

### Table 7: Test set Pearson Correlation from Random Forest classifier

| | KDA* | Others+KDA* | Others |
|---|---|---|---|
| Likert | 0.33 (2nd) | **0.42** | 0.21 |
| Accept | 0.41 (2nd) | **0.49** | 0.19 |
| Irrelevancy | 0.22 (2nd) | **0.23** | -0.03 |
| Low Readability | **-0.17** | -0.19 | **-0.17** |
| Multi-Ans. | 0.21 (2nd) | **0.35** | 0.11 |
| Wrong Ans. | **0.22** | 0.18 (2nd) | -0.18 |

*"Others" = n-gram metrics (BLEU, ROUGE, METEOR); "KDA\*" = both KDA_cont and KDA_disc combined as inputs. Bold = best result, "(2nd)" = second-best (underlined in original). Results averaged across 10 trials of 4-fold stratified cross-validation.*

**Key Takeaway:** A Random Forest classifier trained on KDA_disc + KDA_cont substantially outperforms one trained on n-gram metrics for predicting expert quality labels (Likert, Accept, Irrelevancy, Multiple-Answers, Wrong-Answer) — except for "Low Readability," where KDA metrics provide no meaningful signal (since PLMs, unlike students, aren't sensitive to jargon/vocabulary difficulty). Combining KDA with n-gram metrics gives the best overall performance for most measures.

---

### Table 8: Sample questions where KDA agrees or disagrees with Expert Likert scale

| # | Model | Dataset | Question | Fact | Options (✓=correct) | KDA_cont | KDA_disc | Likert | BLEU |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Human | TabMCQ | Where can Coyoteite be found? | Coyoteite can be found in rocks | rocks✓, scissors, spoons, pens | 0.90 | 1.00 | 3.29 | 100 |
| 1 | T5DG | TabMCQ | Where can Coyoteite be found? | Coyoteite can be found in rocks | rocks✓, plastic, wood, glass | 0.90 | 1.00 | 3.71 | 0 |
| 2 | Human | TabMCQ | A cave is formed by _. | A(n) cave is formed by weathering | weathering✓, glacial erosion, plate tectonics, continental drift | 0.77 | 0.87 | 3.00 | 100 |
| 2 | QDG | TabMCQ | A cave is formed by what process? | A(n) cave is formed by weathering | weathering✓, glaciers, volcanoes, erosion | 0.82 | 1.00 | 3.29 | 62.5† |
| 3 | KDDG | OBQA | Quartz crystals are made up of | a quartz is made of six-sided transparent crystals | hexagons✓, oval, square, sphere | 0.39 | 0.38 | 3.00 | 0 |
| 4 | KDDG | SciQ | Assume a molecule must cross a plasma membrane into what? | Assume a molecule must cross the plasma membrane into a cell... (excerpt) | cell✓, electron, tissue, plasma | 0.81 | 0.93 | 1.71 | 0 |

*†: For QDG, BLEU for the question stem was evaluated. Rows 1–2 = "KDA and Experts Agree"; Rows 3–4 = "KDA and Experts Disagree."*

**Key Takeaway:** Cases #1–2 show KDA_cont correctly assigning high scores to novel, non-gold-matching distractors (e.g., "wood") that BLEU penalizes unfairly. Case #3 shows a failure mode where the question requires external commonsense reasoning ("hexagon has six sides") not stated in the fact, so PLMs (unlike humans) score it low despite experts approving it. Case #4 shows the opposite failure — PLMs can't judge that a question tests a "mere assumption" rather than a meaningful fact, so KDA is high but expert Likert is low.

---

### Table 9: Cohen's kappa coefficient for inter-annotator agreement

| | obqa | tabMCQ | sciQ | human | qg+dg | dg | kddg | All |
|---|---|---|---|---|---|---|---|---|
| kappa | 0.18 | 0.07 | 0.31 | 0.09 | 0.23 | 0.24 | 0.22 | 0.20 |

*All 21 pairwise coefficients between 7 annotators were averaged.*

**Key Takeaway:** Inter-rater agreement among the 7 expert teachers was generally low (avg. κ = 0.20), reflecting the subjectivity of "good MCQ" judgments — though agreement was notably higher (κ > 0.3–0.4) specifically for identifying "bad" MCQs (those with low KDA_cont).

---

### Table 10 (Appendix): Correlation of KDA_cont and Human Likert Score per generation model

| | OBQA | TabMCQ | SciQ | All |
|---|---|---|---|---|
| Human | -0.57 | -0.18 | 0.06 | -0.01 |
| KDDG | 0.01 | 0.13 | 0.75\* | 0.42\* |
| DG | 0.59 | -0.01 | 0.65 | 0.61\*\* |
| QDG | 0.13 | 0.26 | 0.58 | 0.64\*\* |
| DGen models | 0.32 | 0.16 | 0.65\*\* | 0.51\*\* |

*"DGen models" aggregates both KDDG and DG models to assess distractor-generation evaluation specifically.*

**Key Takeaway:** Correlation between KDA_cont and expert Likert scores varies substantially by which generation model produced the question — notably, correlation for purely human-authored questions is weak or even negative, while model-generated questions (DG, QDG) show much stronger positive correlation, suggesting KDA_cont is particularly effective at evaluating machine-generated content.

---

### Table 11 (Appendix): Average Correctness of Datasets

| | avg. Rq | avg. Rf | avg. Rq+f |
|---|---|---|---|
| OBQA | 0.58 | 0.80 | 0.71 |
| TabMCQ | 0.53 | 0.42 | 0.99 |
| SciQ | 0.56 | 0.42 | 0.96 |

**Key Takeaway:** TabMCQ and SciQ show near-ceiling correctness (0.96–0.99) once the fact is shown, confirming these datasets contain "too easy" questions that are simple paraphrases — explaining their weaker KDA-vs-human correlation seen in Table 4.

---

### Table 12 (Appendix): Two sub-metrics of KDA_cont

| Sub Metric | Model Count (Total Size) | KDA (Valid) | Likert (Test) |
|---|---|---|---|
| KDA_small | 4 (3.5GB) | 0.740 | 0.377 |
| KDA_large | 10 (19.2GB) | 0.784 | 0.421 |

*KDA_small uses T5-cbqa-small, ALBert-xl, MPNet, SciBERT. KDA_large uses T5-cbqa-small, T5-cbqa-large, ALBert-xl, MPNet, SciBERT, bert-base, BioBERT-base, RoBERTa-base, RoBERTa-large, XLNet-large.*

**Key Takeaway:** Provides lightweight, compute-efficient alternatives to the full 18-model KDA_cont for practitioners with limited resources, at a modest cost in correlation strength (0.740 vs 0.784 with human KDA).

---

### Table 13 (Appendix): Cases of KDA_cont < 0.4 and human Likert > 2.5

| | Eg1 | Eg2 | Eg3 | Eg4 |
|---|---|---|---|---|
| Dataset | OBQA | OBQA | OBQA | OBQA |
| QG model | KDDG | KDDG | KDDG | DG |
| Fact | a beach ball contains gas | water is in the solid state, called ice, for temperatures between 0 and 0 F | friction acts to counter the motion of two objects when their surfaces are touching | friction acts to counter the motion of two objects when their surfaces are touching |
| Question | Which would you likely find inside a beach ball? | Global warming is lowering the world's amount of | When it's flying, a plane has no friction with the | When it's flying, a plane has no friction with the |
| Answer | air | ice | ground | ground |
| Options | food, gas, water | snow, water, air | power, air, water | air, water, sky |
| Gold options | steam, water, cheese | hurricanes, carbon dioxide, ocean levels | wings, clouds, air | wings, clouds, air |
| Likert | 2.57 | 2.57 | 2.71 | 3.0 |
| KDA_cont | 0.11 | 0.38 | 0.27 | 0.29 |
| KDA_disc | 0.08 | 0.36 | 0.40 | 0.54 |
| BLEU | 33.3 | 0.00 | 33.3 | 33.3 |

**Key Takeaway:** These are "Low KDA, High Likert" failure cases (Section 4.4 case study, e.g. Eg3/Eg4 relate to the "hexagon" case discussed in text) where questions require multi-step reasoning or outside commonsense knowledge not explicit in the fact — PLM solvers underperform relative to humans, deflating KDA scores despite experts finding the questions acceptable.

---

### Table 14 (Appendix): Cases of KDA_cont > 0.8 and human Likert < 2.5

| | Eg1 | Eg2 | Eg3 | Eg4 |
|---|---|---|---|---|
| Dataset | TabMCQ | TabMCQ | TabMCQ | TabMCQ |
| QG model | QG+DG | QG+DG | DG | DG |
| Fact | water is an insulator of electricity | air is an insulator of electricity | warm is a term that can describe air temperature | water is an insulator of electricity |
| Question | Water is an insulator of what? | Air is an insulator of what? | What does the term warm describe? | Water is an insulator of what? |
| Answer | electricity | electricity | air temperature | electricity |
| Options | cold, heat, warmth | heat, cold, warmth | precipitation, wind speed, cloud cover | heat, warmth, cold |
| Gold options | wind, air, heat | heat, water, metal | wind speed, air pressure, optical phenomenon | wind, air, heat |
| Likert | 2.14 | 2.42 | 2.29 | 2.00 |
| KDA_cont | 0.84 | 0.82 | 0.84 | 0.84 |
| KDA_disc | 0.93 | 0.93 | 0.86 | 0.93 |
| BLEU | 33.33 | 33.33 | 33.33 | 33.33 |

**Key Takeaway:** These are "High KDA, Low Likert" failure cases — questions that are trivially answerable (near-paraphrases of the fact) score high on KDA but are judged low-value by experts because they test superficial recall rather than meaningful understanding, or (per Section 4.4) test a "mere assumption" rather than substantive knowledge.

---

## 3. Figure Catalog & In-Depth Explanation

### Figure 1: Problem formulation
**Caption:** *"A multiple-choice question designed to test the knowledge of a target fact must satisfy the following criterion: a question has to be answered by only a student who knows the fact, not a student who doesn't."*

**Description & Insights:** This figure contrasts a "bad question (too easy)" against a "good question." In the bad-question example ("Predators eat: (a) lions (b) humans (c) bunnies (d) grass"), the student answers correctly ("bunnies") regardless of whether the fact ("Predators eat prey") is shown — indicated by a green checkmark in both the "Q only" and "F+Q" (fact + question) conditions. This demonstrates the question doesn't actually test knowledge of the fact; it can be answered via general reasoning/guessing alone. In the good-question example ("Where can Coyoteite be found?"), the student answers incorrectly ("plastic," red X) without the fact but answers correctly ("rocks," green check) once shown the fact ("Coyoteite can be found in rocks"). This is the visual grounding for the entire KDA concept: answerability should be *conditional* on fact knowledge, not independent of it.

---

### Figure 2: Flow diagram of human experiment and model inference
**Description & Insights:** Split into two panels: **[Human]** (left) and **[Model]** (right).
- **Human panel:** Shows participant $H_j$ answering a question "Without fact knowledge" (getting $r_j^q = 0$, i.e., wrong) and then, after being shown the fact, answering again "With fact knowledge" (getting $r_j^{q+f} = 1$, i.e., correct). This mirrors the two-phase design of the actual survey: each phase tests 40 facts, and after solving all questions without the fact, each student re-solves all questions with the corresponding fact shown.
- **Model panel:** Shows the language model $LM_k$ being run twice — once with a bare prompt ("Predators eat") producing $P(R_k^q=1) = 0.32$ via softmax over the answer options (values like 0.1, 0.2, 0.5, 0.1 → normalized to 0.22/0.24/0.32/0.22), and once with the fact prepended ("Predators eat prey <s> Predators eat") producing $P(R_k^{q+f}=1) = 0.36$ (raw logits 0.05/0.1/0.6/0.05 → normalized 0.21/0.22/0.36/0.21).

This figure is the operational blueprint for both human KDA measurement (Eq. 1) and the PLM-based KDA_disc/KDA_cont approximations (Eqs. 2–3), showing exactly how "with fact" vs. "without fact" answer probabilities are obtained in each setting.

---

### Figure 3: Scatter plot of KDA_cont vs. human-measured KDA
**Description & Insights:** X-axis is KDA_cont (PLM-based "soft" KDA); Y-axis is KDA (gold, human-measured knowledge dependency). Points are colored by dataset: SciQ, OBQA, TabMCQ. The overall Pearson correlation between the two metrics is reported as 0.74. Visually, the plot shows a strong positive trend, with data clustering toward the top-right for high-quality questions. Notably, many SciQ and OBQA points cluster near $y=1.0$ (perfect human answerability once given the fact), which the text uses to explain why correlation is comparatively weaker for those "too easy" datasets — there's less variance in human KDA to correlate against, even though KDA_cont varies more broadly.

---

### Figure 4: Cumulative acceptance-rate graph
**Caption:** *"Cumulative graph showing how the acceptance rate of questions above a specific KDA_cont value changes."*

**Description & Insights:** X-axis is a KDA_cont threshold value (0.1–1.0); Y-axis is "Acceptance Rate" — the proportion of questions scoring **above** that threshold that were accepted by expert teachers for classroom use. The curve is monotonically increasing, starting around ~0.55–0.6 at low thresholds and approaching 1.0 as the threshold nears 1.0. The text highlights that 61%, 51%, 39%, and 19% of all questions score above KDA_cont thresholds of 0.6, 0.7, 0.8, and 0.9 respectively (confirming sufficient sample sizes per bin), and that 82% of questions scoring KDA_cont > 0.8 were accepted for classroom use — demonstrating KDA_cont's value as a practical filtering threshold for deploying generated MCQs.

---

### Figure 5: Regression graph of expert flaw labels vs. KDA_cont
**Description & Insights:** X-axis is KDA_cont; Y-axis is the number of flaw labels assigned by experts to a question (out of: low readability, multiple answers, wrong answer, irrelevant), for each specific KDA_cont value, with four separate regression lines (one per flaw type) plus 95% confidence interval shading. Three of the four lines — "multiple answers," "wrong answer," and "irrelevant" — trend clearly **downward** as KDA_cont increases, indicating that higher KDA_cont scores predict fewer of these flaws. The "low readability" line stays comparatively flat/slightly upward, visually confirming the paper's claim (Section 4.3) that KDA_cont has essentially no predictive power over question readability, since PLMs don't experience vocabulary/jargon difficulty the way human students do.

---

## 4. Existing Limitations

The authors explicitly enumerate five limitations (Section 6):

1. **Prompt Can Bias Solvers' Decision (6.1):** The prompt-based fact-injection method can bias PLM "students" toward the single labeled answer in cases where a question legitimately has multiple valid answers (e.g., "Select an option that is in a liquid state at 20°C" with both "water" and "orange juice" being valid, but only "water" labeled). This means KDA cannot reliably filter out such improperly-labeled low-quality questions.

2. **Too Easy Questions (6.2):** The core assumption — that a well-designed MCQ is answerable *iff* the student knows the fact — breaks down for questions answerable via the question stem alone or by ruling out topically irrelevant distractors, regardless of actual fact knowledge. This means some "knowledgeable" classifications in the human study may not reflect true target-fact knowledge.

3. **Difficulty of Measuring PLM's Ignorance (6.3):** Because the authors lack access to the PLMs' training corpora, they cannot guarantee a given PLM doesn't already "know" a fact from pretraining (contamination), which would confound the "without fact" baseline condition. The authors flag this as future work — noting a need for language models with reading comprehension ability but *without* prior factual knowledge.

4. **Low Agreement between Teachers (6.4):** Cohen's kappa inter-rater agreement among the 7 expert annotators averaged only 0.20, attributed to subjective differing views on "good" MCQs. Agreement was notably better (κ > 0.3–0.4) specifically for identifying clearly "bad" questions (KDA_cont < 0.3 / < 0.2).

5. **Availability of PLMs for Low-Resource Languages (6.5):** KDA_cont/KDA_disc have only been validated on English questions using English-trained PLMs; because the metric depends on PLMs with strong reading comprehension, applicability to low-resource languages (where such PLMs may not exist) is limited.

**Additional technical constraints noted elsewhere in the paper:**
- **Compute cost:** Training all 18 solver PLMs on Google Cloud cost approximately $14,000 (on-demand, non-preemptive pricing), which the authors partially mitigate via the smaller KDA_small/KDA_large sub-metric variants (Table 12).
- **Readability blind spot:** Table 7 and Figure 5 both empirically confirm KDA_cont/KDA_disc show no predictive power for the "Low Readability" quality dimension.
- **Failure modes documented in Section 4.4 / Tables 13–14:** (a) "Low KDA, High Likert" — questions requiring extra reasoning/commonsense beyond the stated fact are unfairly penalized by PLM solvers; (b) "High KDA, Low Likert" — trivial paraphrase questions, or questions testing a mere assumption rather than substantive content (notably common in the crowd-sourced SciQ dataset), score artificially high since PLMs can't judge educational *meaningfulness* of the underlying fact.
- **Ethical consideration (Section 7):** The authors note that questions passing the KDA filter may still contain other issues such as gender or racial bias, since the metric is narrowly scoped to knowledge dependency and does not screen for fairness/bias.

---

## 5. Conclusion & Future Outlook

**High-Level Conclusion:** The paper introduces **Knowledge Dependent Answerability (KDA)**, a reference-free evaluation criterion formalizing the intuitive requirement that a good MCQ should be answerable specifically *because* the student knows the target fact (not for unrelated reasons). Because directly measuring KDA requires costly human trials, the authors propose two automatic surrogates, **KDA_disc** and **KDA_cont**, which substitute an ensemble of pre-trained language models for human solvers, comparing answer correctness/probability with and without the target fact prompted. Across three real-world MCQ datasets (OBQA, TabMCQ, SciQ), both automatic metrics show strong correlation with human-measured KDA (up to 0.80 for KDA_disc, aggregated across datasets) and with expert-teacher Likert usability ratings — substantially outperforming traditional n-gram similarity metrics (BLEU, ROUGE, METEOR), which correlate only weakly (0.21–0.27). Furthermore, combining KDA metrics with n-gram similarity metrics yields even stronger predictive power (up to 0.49 Pearson correlation) for expert-labeled MCQ quality dimensions like overall acceptance.

**Future Work / Extensions (explicitly suggested by authors):**
- Expanding KDA's applicability beyond MCQs to other assessment question formats, such as **short-answer questions** or **multi-hop reasoning questions**.
- Developing solver models that can better disentangle "reading comprehension ability" from "prior factual knowledge," to address the PLM-ignorance measurement problem (Limitation 6.3).
- Using larger/better-reasoning language models as solvers to improve correlation on more complex datasets like OBQA (as hinted by the monotonic improvement trend in Table 5).
- Extending validation to low-resource languages once suitable PLMs become available.

**Practical Implications:** KDA_cont and KDA_disc are proposed as practical, scalable **filters** for automatic question generation pipelines — for instance, the paper shows 82% of questions scoring KDA_cont > 0.8 were independently accepted by expert teachers for classroom use (Figure 4), suggesting the metric could be deployed as a lightweight quality gate before human review, substantially reducing the manual burden of vetting AI-generated educational assessment content. The authors released their code and model weights publicly to facilitate adoption.

---

*Note: This paper contains citations/references (e.g., Papineni et al. 2002 for BLEU, Lin 2004 for ROUGE, Banerjee & Lavie 2005 for METEOR, Raffel et al. 2020 for T5, Roberts et al. 2020 for closed-book QA, etc.) which have been transcribed directly from the source PDF's reference list and are not independently verified or hallucinated by this summarization process.*
