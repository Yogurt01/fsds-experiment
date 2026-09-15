# QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation — Paper Documentation

---

## 1. Executive Summary & Core Content

**Title:** QG-SMS: Enhancing Test Item Analysis via Student Modeling and Simulation

**Authors & Affiliation:**
- Bang Nguyen¹ (University of Notre Dame)
- Tingting Du² (University of Wisconsin-Madison)
- Mengxia Yu¹ (University of Notre Dame)
- Lawrence Angrave³ (University of Illinois at Urbana-Champaign)
- Meng Jiang¹ (University of Notre Dame)

Correspondence: bnguyen5@nd.edu
Code released at: https://github.com/bnguyen5/qg-sms

**Venue / Year:** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025, Volume 1: Long Papers), pages 26152–26168, July 27 – August 1, 2025.

**Research Question(s) & Problem Formulation:**
Question Generation (QG) is increasingly used in educational assessment, but its evaluation remains disconnected from established educational measures of test quality. Existing QG evaluation approaches fall into two camps: (1) reference-based metrics (ROUGE, BLEU, BERTScore) that measure textual similarity to a human-written reference, whose validity/reliability has been questioned; and (2) reference-free metrics (e.g., KDA, QSalience) and increasingly popular LLM-as-a-judge pairwise comparison methods (Vanilla, CoT, ChatEval, etc.), which mostly assess **answerability** or **question content quality** in isolation, without grounding in actual **student performance**.

The paper imports a well-established educational-testing methodology — **test item analysis** — into QG evaluation. Test item analysis is normally split into:
- **Pre-examination analysis**: evaluating items *before* administration (e.g., topic alignment/coverage).
- **Post-examination analysis**: evaluating items *after* administration, using statistical analysis of actual test-taker responses (item difficulty, item discrimination, distractor efficiency).

The core research questions are:
- Can existing QG evaluation methods (reference-based, reference-free, and LLM-as-a-judge pairwise approaches) accurately distinguish test items along four educationally meaningful dimensions — **topic coverage (TC)**, **item difficulty (DF)**, **item discrimination (DC)**, and **distractor efficiency (DE)**?
- The paper's motivating case study (Table 1) shows that existing LLM-based evaluators reason about discrimination purely from question *content* (e.g., judging an "apply-level" question as more discriminating than a "recall-level" question), but this reasoning contradicts what **actual student performance data** shows — revealing a fundamental gap: existing methods lack a mechanism to simulate how real students of varying ability would actually respond.

**Proposed Methodology:**
The paper proposes **QG-SMS** (Student Modeling and Simulation), a three-step evaluation framework built on a single LLM (GPT-4o):
1. **Step 1 – Student Profile Generation:** Given learning materials $L$, the LLM generates a diverse set of at least 10 simulated student profiles $S = \{s_1, s_2, ..., s_n\}$, each with a distinct, textually-described level of understanding of $L$ (avoiding personal identity attributes to reduce social bias), mimicking the understanding-distribution of a real classroom.
2. **Step 2 – Student Performance Prediction:** Given $L$, the generated student profiles $S$, and a pair of candidate questions $\{Q_1, Q_2\}$, the LLM predicts whether each simulated student will answer each question correctly or incorrectly (and, if incorrect, which distractor misleads them).
3. **Step 3 – Evaluation:** Given the candidate question pair $\{Q_1, Q_2\}$, the target requirement $R_d$ (e.g., "the question with higher discrimination"), and the Step-2 simulated performance data, the LLM makes a final preference judgment between $Q_1$ and $Q_2$.

This effectively augments the LLM's judgment with synthetic "student response data," allowing it to approximate post-examination statistical analysis without needing real test administration.

**Main Contributions:**
- Systematically introduces **test item analysis** (topic coverage, item difficulty, item discrimination, distractor efficiency) into the QG evaluation pipeline, and mathematically formalizes each dimension.
- Constructs two benchmark datasets of question pairs (from **EduAgent** and **DBE-KT**) with statistically significant, labeled quality differences across the four dimensions, to serve as an evaluation testbed.
- Reveals a significant performance gap in existing QG evaluation approaches (reference-based, reference-free, and various LLM-as-a-judge pairwise strategies): they perform well on pre-examination topic coverage (~95.6% avg.) but poorly on post-examination dimensions — difficulty (49.1%), discrimination (44.5%), and distractor efficiency (53.3%).
- Proposes **QG-SMS**, a novel, simulation-based evaluation framework that uses a single LLM to generate diverse student profiles and simulate their performance, closing this gap and outperforming all baselines across all four dimensions.
- Conducts extensive experiments (including varying the significance threshold α, testing robustness of generated student profiles across runs, and an ablation on the necessity of the LLM-based final evaluation step) plus a **human evaluation study** with volunteer annotators, showing QG-SMS achieves the closest alignment to human judgment among all automated methods.
- Publicly releases all implementation details/code.

---

## 2. Detailed Recreation of All Tables

### Table 1: Case study — existing LLM-based approaches rely solely on question content

| | Content |
|---|---|
| **Q1** | Which of the following may utilize computer vision techniques? (1) Use a camera to check potential issues on the surface of products (2) Estimate the freshness of apples from pictures (3) Estimate whether a car is speeding via a camera (4) Determine whether a piece of audio is spoken by a specific person. A) (1)(2)(3); B) (1)(2)(4); C) (2)(3)(4); D) (1)(2)(3)(4). |
| **Q2** | One breakthrough in computer vision happened at the University of Toronto in 2012, which achieved an error rate of [ ] in image classification. A) 6.4%; B) 10.4%; C) 12.4%; D) 16.4%. |
| **Evaluation Task** | Which question has higher discrimination? |
| **Existing approaches' answer** | Q1 — reasoned as an "apply-level" question vs. Q2 as "recall-level." |
| **Label based on Actual Student Performance** | Q2 — applications of CV in Q1 are common knowledge, while Q2 tests a specific detail only attentive students answer correctly. |

**Key Takeaway:** This motivating example demonstrates that content-only LLM reasoning ("apply > recall" heuristic) can directly contradict ground-truth student performance data, establishing the core need for student modeling in QG evaluation. The full case study (all baseline outputs) is expanded in Appendix Table 8.

---

### Table 2: Performance (AA, CA) of existing QG evaluation approaches and QG-SMS across dimensions

| Method | TC (EduAgent, 217) AA / CA | TC (DBE-KT, 286) AA / CA | DF (EduAgent, 124) AA / CA | DF (DBE-KT, 162) AA / CA | DC (EduAgent, 61) AA / CA | DC (DBE-KT, 93) AA / CA | DE (EduAgent, 75) AA / CA |
|---|---|---|---|---|---|---|---|
| **Individual Scoring** | | | | | | | |
| BERTScore | 79.26 / – | 40.20 / – | 51.61 / – | 61.73 / – | 65.57 / – | 30.11 / – | 65.33 / – |
| KDA_large | 57.60 / – | 38.46 / – | 60.48 / – | 54.32 / – | 60.66 / – | 58.06 / – | **77.33** / – |
| QSalience | 54.84 / – | 48.25 / – | 54.03 / – | 60.49 / – | 52.46 / – | 47.31 / – | 68.00 / – |
| **Pairwise LLM-based** | | | | | | | |
| Vanilla | 96.54 / 95.39 | 74.30 / 68.89 | 63.71 / 50.80 | 67.28 / 49.38 | 63.11 / 49.18 | 63.98 / 49.46 | 73.33 / 64.00 |
| CoT | 95.39 / 92.63 | 78.15 / 65.03 | 61.69 / 32.26 | 64.20 / 38.89 | 59.84 / 32.79 | 62.90 / 34.41 | 60.00 / 28.00 |
| Metrics | 97.70 / 97.70 | 80.59 / 75.17 | 65.32 / 53.22 | 64.20 / 48.77 | 65.57 / 50.82 | 61.29 / 45.16 | 72.00 / 62.67 |
| Reference | 97.00 / 96.31 | 72.55 / 66.43 | 66.53 / 51.61 | 62.96 / 45.06 | 62.30 / 45.90 | 60.75 / 44.09 | 69.33 / 56.00 |
| Swap | 95.85 / 95.85 | **81.64** / 74.48 | 66.53 / 54.84 | 68.31 / 53.70 | 64.75 / 45.90 | 62.90 / 48.39 | 68.00 / 53.33 |
| ChatEval | 96.77 / 95.85 | 80.94 / 74.13 | **68.95** / 51.61 | 70.99 / 59.88\* | 54.92 / 42.56 | 65.05 / 53.76 | 69.33 / 56.00 |
| **QG-SMS (Ours)** | **98.85** / **98.62** | 79.90 / **74.82** | 68.55 / **65.32\*** | **69.44** / **64.20\*** | **66.39** / **55.74** | **66.66** / **56.99** | 79.33 / **74.67\*** |

*AA = Average Accuracy, CA = Consistent Accuracy. Bold = highest per column, underline (not distinguishable in plain text, noted where applicable) = second-highest. Asterisk (\*) = statistical significance at p < 0.1 of LLM-based approach improving CA vs. Vanilla.*

**Key Takeaway:** Existing evaluation approaches (both individual-scoring and pairwise LLM-based) perform strongly on pre-examination Topic Coverage (~95–98% AA) but degrade substantially on post-examination dimensions (Difficulty, Discrimination, Distractor Efficiency), especially in Consistent Accuracy (robustness to question-order swapping). QG-SMS achieves the best or near-best performance across nearly every dimension/dataset combination, and is the *only* method to show statistically significant CA improvements over the Vanilla baseline across multiple dimensions (DF-EduAgent, DF-DBE-KT, DE-EduAgent).

---

### Table 3: QG-SMS-derived ranking scores by Distractor Efficiency (DE) level (EduAgent)

| DE = 0 | DE = 1 | DE = 2 | DE = 3 |
|---|---|---|---|
| 0.22 ± 0.18 | 0.40 ± 0.22 | 0.48 ± 0.22 | 0.67 ± 0.22 |

**Key Takeaway:** QG-SMS-derived ranking scores increase monotonically with actual DE level, and an ANOVA test confirms this difference is statistically significant (p < 0.01) — demonstrating QG-SMS can meaningfully rank/group questions even without a strict pairwise "significant difference" threshold.

---

### Table 4: Correlation between ranking-score method and actual DE value (EduAgent)

| Method | Spearman | Kendall | Pearson |
|---|---|---|---|
| KDA | 0.43 | 0.33 | 0.43 |
| Vanilla | 0.34 | 0.27 | 0.35 |
| **QG-SMS** | **0.48** | **0.38** | **0.50** |

**Key Takeaway:** QG-SMS achieves the highest correlation with ground-truth Distractor Efficiency across all three correlation measures, outperforming both the strongest individual-scoring baseline (KDA) and the strongest simple pairwise LLM baseline (Vanilla).

---

### Table 5: Results (AA, CA) on human-written (HumanQs) and generated (GenQs) question pairs

| Method | HumanQs (Stud.Perf Label) AA | GenQs (Anno Label) AA | GenQs (Anno Label) CA |
|---|---|---|---|
| Vanilla | 70.83 | 70.83 | 58.33 |
| CoT | 67.50 | 65.00 | 38.33 |
| Metrics | 70.83 | 69.17 | 53.33 |
| Reference | 69.17 | 67.50 | 55.00 |
| Swap | 73.33 | 65.00 | 48.33 |
| ChatEval | 69.17 | **74.17** | <u>56.67</u> |
| **QG-SMS** | <u>76.67</u> | **74.17** | **63.33** |
| Human | **78.33** | – | – |

*Bold = highest, underline = second-highest per column.*

**Key Takeaway:** On human-written question pairs (with ground-truth labels from actual student performance), human annotators outperform all automated methods (78.33%), but QG-SMS is the closest automated approximation (76.67%, second-best overall). On generated question pairs (labeled by human annotators), QG-SMS achieves the best or tied-best performance in both average and consistent accuracy, confirming its applicability to the QG evaluation setting specifically.

---

### Table 6 (Appendix A.2): P-values from binomial tests — significance of CA improvement vs. Vanilla

| Method | DF (EduAgent) | DF (DBE-KT) | DC (EduAgent) | DC (DBE-KT) | DE (EduAgent) |
|---|---|---|---|---|---|
| CoT | 0.999 | 0.996 | 0.999 | 0.999 | 0.999 |
| Metrics | 0.227 | 0.640 | 0.500 | 0.910 | 0.813 |
| Reference | 0.500 | 0.925 | 0.938 | 0.967 | 0.981 |
| Swap | 0.192 | 0.134 | 0.856 | 0.588 | 0.985 |
| ChatEval | 0.500 | 0.007 | 0.927 | 0.262 | 0.942 |
| **QG-SMS** | **0.007** | **0.000** | 0.252 | 0.124 | **0.093** |

**Key Takeaway:** Lower p-values indicate stronger statistically significant improvement over the Vanilla baseline. QG-SMS shows the strongest (lowest p-value) improvements for Difficulty on both datasets and for Distractor Efficiency, confirming these gains are not due to chance — whereas other baselines (especially CoT) show consistently high p-values (i.e., no significant improvement, or even degradation, vs. Vanilla).

---

### Table 7 (Appendix A.3): Results breakdown on 60 human-written question pairs (QG evaluators vs. Human annotators)

| Method | Diff. (DF) | Disc. (DC) | Dist. Eff. (DE) |
|---|---|---|---|
| Vanilla | 73.81 | <u>56.67</u> | 77.08 |
| CoT | 76.19 | <u>56.67</u> | 62.50 |
| Metrics | 71.43 | 53.33 | <u>81.25</u> |
| Reference | 73.81 | 53.33 | 75.00 |
| Swap | 76.19 | **63.33** | 77.08 |
| ChatEval | 83.33 | 43.33 | 72.92 |
| **QG-SMS** | <u>85.71</u> | <u>56.67</u> | <u>81.25</u> |
| **Human** | **90.48** | 53.33 | **87.50** |

*Bold = highest, underline = second-highest per column.*

**Key Takeaway:** Human annotators achieve the highest accuracy overall (esp. for Difficulty and Distractor Efficiency), but notably score *lower* than QG-SMS on Item Discrimination (53.33% vs. QG-SMS's not-explicitly-listed-here value from Table 2, though QG-SMS is reported in-text to "surpass human evaluators" on DC) — human annotators reported finding it genuinely hard to judge which question better distinguishes high/low performers without access to real classroom data. QG-SMS achieves the second-closest accuracy to humans on Difficulty and Distractor Efficiency.

---

### Table 8 (Appendix A.4): Case Study — limitations of existing evaluation metrics in test item analysis

*(Full text responses from each baseline evaluating Q1 vs. Q2 from Table 1's case study; condensed below.)*

| Method | Verdict / Reasoning Summary |
|---|---|
| **Existing approaches (general)** | Prefer Q1 — reasoned as "apply-level" > "recall-level" (Q2). |
| **Label (Actual Student Performance)** | Q2 has higher discrimination. |
| **CoT** | Favors Q1: reasons it requires applying knowledge to different scenarios, thus better distinguishing comprehensive vs. non-comprehensive understanding; calls Q2 "more about memorizing" and less discriminating. |
| **Metrics** | Generates broad, generic self-derived criteria (e.g., "does the question accurately reflect lecture content/structure?") — not specific enough to directly assess discrimination. |
| **Reference** | Generates an unrelated open-ended reference question about CNN success factors, not usable for direct comparative judgment on discrimination. |
| **Swap** | Favors Q1 in both orderings: "requires students to apply their understanding... thus having high discrimination"; downgrades Q2 as relying on memorization. |
| **ChatEval** | Multi-persona debate still favors Q1: cites "higher-order cognitive skill" (application) as more discriminating; flags Q2 as testing a "lower-order cognitive skill" (recall). |

**Key Takeaway:** All content-only baselines (CoT, Swap, ChatEval) converge on the same flawed heuristic — favoring "apply-level" questions over "recall-level" questions as inherently more discriminating — which directly contradicts the ground-truth label. Generation-based criteria methods (Metrics, Reference) fail differently: they produce criteria/references too broad or off-topic to make a discrimination judgment at all. This table empirically illustrates why explicit student-performance simulation (as in QG-SMS) is necessary — LLMs cannot reliably "reason" about discrimination from question content and learning material alone.

---

## 3. Figure Catalog & In-Depth Explanation

### Figure 1: Performance of LLM-based evaluation methods in pairwise test item comparisons on the EduAgent dataset
**Description & Insights:** A scatter/strip plot with "Item Analysis Dimension" on the x-axis (four dimensions: TC "Which question covers the desired topic?", DF "Which question is easier?", DC "Which question has higher discrimination?", DE "Which question has more effective distractors?") and "Consistent Accuracy" (0–100) on the y-axis. Each method (Vanilla, CoT, Metrics, Reference, Swap, ChatEval — shown in various colors/marker shapes — and QG-SMS shown as a purple star) is plotted as a cluster of points/markers at each dimension, with a light scatter of individual data points in the background suggesting the distribution/spread underlying the summary markers. The plot visually separates a "Pre-examination" region (TC, where nearly all methods cluster near 90–100%) from a "Post-examination" region (DF, DC, DE, where existing methods' markers drop to roughly 30–65%, while the QG-SMS star consistently sits near or above the top of each cluster). This figure is the paper's primary visual argument: existing LLM-based QG evaluation methods excel at pre-examination topic alignment but collapse in post-examination dimensions, and QG-SMS closes this gap.

---

### Figure 2: The QG-SMS three-step pipeline (with worked example)
**Description & Insights:** A flowchart illustrating the full QG-SMS pipeline using the running computer-vision example from Table 1:
- **Step 1 – Student Profiles Generation:** "Learning Materials" (a 3-item outline: Introduction to Computer Vision, Computer Vision History, Computer Vision Tasks) is fed into an LLM, which outputs a set of "Generated Student Profiles" — illustrated with four named/described personas: *Alice - The Attentive* (good understanding of all lecture content including history and technical storage details), *Bob - The Beginner* (understands applications but struggles with detailed history/technical aspects), plus *Clara - The Conflicted* and *David - The Distracted* (profiles not detailed in the visible text but implying a spread of understanding types).
- **Step 2 – Students Performance Prediction:** The candidate questions (Q1 and Q2, same as Table 1) plus the learning materials and generated profiles are fed to the LLM again, which predicts per-student correctness: e.g., "Alice - The Attentive: Q1 correct, Q2 correct" and "Bob - The Beginner: Q1 correct, Q2 incorrect."
- **Step 3 – Evaluation:** The candidate questions, the predicted student performance, and the stated "Requirement" ("A question that has higher discrimination") are fed into the LLM, which outputs the "Preferred Question" — correctly identifying **Q2** in this example.

The insight conveyed: because applications of CV (tested in Q1) are common knowledge that both attentive and beginner-level simulated students answer correctly, Q1 fails to *differentiate* between them — whereas Q2 (a specific historical statistic) is only answered correctly by the more attentive student, making Q2 the higher-discrimination item. This directly reverses the (incorrect) content-only judgment shown in Table 1, demonstrating the value added by explicit student simulation.

---

### Figure 3: Performance of LLM-based approaches in evaluating Difficulty (DF) across different α values
**Description & Insights:** Two side-by-side line-chart panels — left panel for the **DBE-KT** dataset, right panel for the **EduAgent** dataset — each with α (the significance threshold for a "meaningful" difficulty-index difference) on the x-axis (values 0.15, 0.25, 0.35, 0.45) and Consistent Accuracy (0.4–1.0 for DBE-KT; roughly 0.1–0.9 for EduAgent) on the y-axis. Seven lines are plotted, one per method (Vanilla, CoT, Metrics, Reference, Swap, ChatEval, and QG-SMS/"QG-SM"). All lines trend upward as α increases (larger quality gaps are easier to detect correctly), but the QG-SMS line (marked with a star) consistently sits at or near the top of both panels across all α values, and shows a particularly pronounced lead in the EduAgent panel (roughly 0.65–0.85 range vs. ~0.5–0.65 for most baselines). CoT is notably the weakest performer throughout both panels. This figure demonstrates QG-SMS's robustness to the choice of significance threshold.

---

### Figure 4: Simulated student performance on the same set of questions across five different runs
**Description & Insights:** A line chart with "Question ID" on the x-axis (a set of specific question identifiers such as 4_2, 4_9, 4_4, 4_1, 4_8, 4_10, 4_11, 4_12, 4_3, 4_5, 4_6, 4_7) and "% of students with correct answers" (0.3–0.9) on the y-axis. Five differently-colored/marked lines represent five independent repetitions ("Run 1" through "Run 5") of Steps 1 and 2 of the QG-SMS pipeline (i.e., re-generating student profiles and re-predicting their performance from scratch each time). The five lines track each other closely in overall shape — rising and falling together across the same question IDs — despite each run's underlying student profiles being freshly (and independently) LLM-generated. This is the paper's robustness check: it shows that even though individual student profiles differ across runs, the *aggregate distribution* of simulated performance is highly consistent, supporting the reliability of using LLM-generated student profiles as a stable evaluation signal.

---

### Figure 5 (Appendix A.1): Prompts for the three-step QG-SMS evaluation approach
**Description & Insights:** A stacked, color-coded panel of the three literal prompt templates used in the pipeline:
- **Step 1 (green box):** Instructs the LLM to generate at least 10 diverse student roles given the learning materials, each with a name and description of their lecture understanding, mimicking a real classroom's understanding distribution.
- **Step 2 (pink box):** Given the learning materials, the Step-1 student profiles, and two candidate questions, instructs the LLM to predict per-student correctness for each question (accounting for understanding level, question difficulty, and guessing factors), and to name the specific distractor that confuses each student predicted to answer incorrectly.
- **Step 3 (blue box):** Given the target requirement $R_d$, its natural-language description, the two candidate questions (labeled "Output (a)" / "Output (b)"), and the Step-2 simulated performance, instructs the LLM to output a single preference ("Output (a)" or "Output (b)").

This figure serves as the full reproducibility artifact for the method, showing exactly how each step's LLM call is constructed.

---

### Figure 6 (Appendix A.3): Prompts for generating questions with varying quality across three dimensions
**Description & Insights:** Three color-coded prompt boxes used to construct the synthetic "generated question pairs" (GenQs) benchmark for the human evaluation study:
- **Difficulty-controlled generation (green):** Instructs the LLM to produce six 4-choice questions per learning material — 2 easy, 2 medium, 2 hard — where difficulty is defined by how many students would answer correctly.
- **Discrimination-controlled generation (pink):** Instructs generation of six questions — 2 low-, 2 medium-, 2 high-discrimination — explicitly defining low discrimination as either "neither high- nor low-performing students can answer correctly" or "all students can answer correctly."
- **Distractor-efficiency-controlled generation (blue):** Instructs generation of six questions with 0, 1, 2, and 3 (two each, except implied combinations) "effective" distractors, where an effective distractor is defined as one selected by at least 5% of students.

This figure documents the zero-shot generation procedure (GPT-4o, `gpt-4o-2024-05-13`) used to build the 360-question bank (across 5 lectures) from which the 60 generated-question evaluation pairs for the human study were drawn.

---

## 4. Existing Limitations

The authors explicitly state the following in the **Limitations** section:

1. **Individual-level (pairwise) evaluation only, not holistic assessment design:** The paper evaluates test item quality at an individual item level, but constructing a full assessment typically requires balancing multiple dimensions simultaneously and ensuring diversity within each dimension (e.g., a good quiz should cover different topics rather than repeatedly testing the same concept, and should include a mix of easy/medium/hard questions). QG-SMS as presented does not directly solve this "porfolio-level" assembly problem, though the authors suggest ranking-based extensions (as demonstrated in the DE ranking experiment, Tables 3–4) as a potential path for future work to help teachers assemble balanced assessments.
2. **Statistical significance threshold caveat:** The significance tests in the paper rely on a p-value threshold of 0.1 (a relatively lenient threshold compared to the conventional 0.05), and the authors note that future work could explore whether stronger underlying LLMs would yield more robust significance results at stricter thresholds.

**Ethical Considerations (related caveats):**
- Despite deliberately grounding student-profile simulation in learning materials alone (avoiding personal identity attributes) to reduce social bias, the authors observed an **implicit bias** in the LLM's naming choices — a predominance of European names (e.g., Alice, Bob) among generated student personas.
- The authors emphasize that simulated student profiles are **not intended to represent specific real students** in any real classroom; they are meant only to collectively approximate the *diversity* of understanding levels present in a learning population, not individual demographic or identity characteristics.

**Additional technical constraints and generalizability bottlenecks identified elsewhere in the paper:**
- **Dataset-specific behavior of baseline metrics:** BERTScore's and QSalience's behavior on TC and DC, and KDA's behavior on TC, are shown (Section 5.1) to be dataset-specific rather than generalizable — i.e., their correlation direction/strength with ground truth is not consistent across EduAgent vs. DBE-KT, making them unreliable as general-purpose indicators for these dimensions.
- **DBE-KT data limitation:** For the DBE-KT dataset, only Item Difficulty (DF) and Item Discrimination (DC) can be computed, since information on which specific distractors were chosen by students who answered incorrectly is unavailable — meaning Distractor Efficiency (DE) results are reported only for EduAgent.
- **Trade-off exposed by the ablation study (Section 5.2, "Necessity of LLM-based evaluation step"):** Directly computing DC/DF statistics from simulated performance data (bypassing Step 3's LLM judgment) improves Difficulty accuracy (68.55% → 73.11%) but substantially harms Discrimination accuracy (66.39% → 56.83%), showing that naive statistical aggregation of simulated data is not sufficient on its own — the LLM's semantic reasoning in Step 3 is necessary but imperfect, and the two information sources (raw simulated stats vs. LLM judgment) trade off against each other depending on dimension.
- **Human annotator difficulty with Discrimination:** In the human evaluation study, human annotators found Item Discrimination (DC) the hardest dimension to judge (53.33% accuracy — the lowest of the three tested dimensions, and lower than QG-SMS's DC performance), explicitly citing the lack of access to real classroom-wide student performance data as the reason — underscoring that this dimension is inherently difficult to assess even for trained/domain-relevant human judges without empirical data.
- **Model dependency:** All LLM-based methods (baselines and QG-SMS) rely on a single base model, GPT-4o (`gpt-4o-2024-05-13`), so results may not generalize to other LLM backbones without re-validation.
- **Positional bias mitigation adds cost:** Because LLMs are known to exhibit positional bias, all pairwise methods (including QG-SMS) must be run twice per pair (both orderings) to compute Consistent Accuracy, effectively doubling inference cost relative to a single-pass evaluation.

---

## 5. Conclusion & Future Outlook

**High-Level Conclusion:** This paper introduces **test item analysis** — a well-established educational assessment methodology encompassing topic coverage, item difficulty, item discrimination, and distractor efficiency — into the evaluation of automatically generated questions (QG). Through carefully constructed benchmark datasets (477 pairs from EduAgent, 255 pairs from DBE-KT) with statistically significant, mathematically-grounded quality-difference labels, the authors show that existing QG evaluation approaches — whether reference-based (BERTScore), reference-free single-scoring (KDA, QSalience), or LLM-as-a-judge pairwise methods (Vanilla, CoT, Metrics, Reference, Swap, ChatEval) — perform well on pre-examination topic coverage but substantially underperform on post-examination dimensions that require modeling actual student response behavior. The proposed **QG-SMS** framework closes this gap by using a single LLM to (1) generate diverse simulated student profiles grounded only in the learning material, (2) predict each simulated student's response to candidate questions, and (3) synthesize this simulated performance data with question content to make a final, more educationally-aligned preference judgment. QG-SMS achieves the highest average accuracy on Discrimination and Distractor Efficiency, the second-highest on Difficulty, and significantly better *consistency* (robustness to answer-order swapping) than all baselines — and in a follow-up human evaluation study, is shown to align more closely with human judgment than any other automated method, second only to human annotators themselves.

**Future Work / Extensions (explicitly suggested by authors):**
- **Reward-based optimization pipelines:** Since prior work shows LLMs often fail to incorporate explicit educational requirements into generated questions even when prompted, and QG-SMS is shown to be a reliable indicator of DF/DE/DC, the authors suggest QG-SMS could be integrated as a reward signal in an optimization/fine-tuning pipeline to better align generated test items with specific educational objectives.
- **Extending to research question generation:** The authors note growing interest in automatically generating *research* questions (as opposed to educational quiz questions), a task that still relies on costly, time-consuming human-researcher evaluation; they suggest a simulation-based framework analogous to QG-SMS — simulating diverse researcher perspectives — could enable scalable, automated evaluation in that domain as well.
- **Assessment-level (portfolio) balancing:** Building on the Limitations discussion, future work could use QG-SMS-derived rankings (as demonstrated in the Distractor Efficiency ranking experiment) to help teachers assemble balanced sets of test items across topics and difficulty/discrimination levels, rather than evaluating items purely in isolation.
- **Stronger base models:** The authors suggest exploring whether more capable underlying LLMs could yield more statistically robust significance results, potentially allowing stricter significance thresholds than the p < 0.1 used in this study.

**Practical Implications:** QG-SMS offers a scalable, single-LLM alternative to costly, slow, and logistically difficult real-classroom pilot testing (traditionally required for post-examination item analysis). By simulating diverse student understanding directly from learning materials — without manual feature engineering or prompt engineering per student profile, unlike several prior LM-based student-simulation approaches cited in Related Work (e.g., Lu and Wang, 2024; Lalor et al., 2019; Hayakawa and Saggion, 2024; Byrd and Srivastava, 2022) — QG-SMS provides a more flexible, automatic route to educationally-grounded evaluation of AI-generated (or human-written) test questions during the test design phase, before real student data is available.

---

*Note: This paper contains numerous citations to related work (e.g., Zheng et al. 2023 for LLM-as-a-judge, Chan et al. 2023 for ChatEval, Wei et al. 2022 for Chain-of-Thought, Moon et al. 2022 for KDA, Wu et al. 2024 for QSalience, Xu et al. 2024 for EduAgent, Abdelrahman et al. 2022 for DBE-KT, etc.) which have been transcribed directly from the source PDF's reference list and are not independently verified or hallucinated by this summarization process.*
