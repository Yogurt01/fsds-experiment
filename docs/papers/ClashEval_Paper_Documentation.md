# ClashEval: Quantifying the Tug-of-War Between an LLM's Internal Prior and External Evidence — Paper Documentation

---

## 1. Executive Summary & Core Content

**Title:** ClashEval: Quantifying the tug-of-war between an LLM's internal prior and external evidence

**Authors & Affiliation:**
- Kevin Wu* (Department of Biomedical Data Science, Stanford University)
- Eric Wu* (Department of Electrical Engineering, Stanford University)
- James Zou (Department of Biomedical Data Science, Stanford University)

*Equal contribution.
Dataset and code: https://github.com/kevinwu23/StanfordClashEval

**Venue / Year:** 38th Conference on Neural Information Processing Systems (NeurIPS 2024), Track on Datasets and Benchmarks.

**Research Question(s) & Problem Formulation:**
Retrieval-Augmented Generation (RAG) is widely used to mitigate LLM hallucination and provide up-to-date knowledge by injecting retrieved documents into the model's context. However, document retrieval is imprecise and can surface erroneous, outdated, or even harmful content (e.g., Google's AI Overview infamously recommending people "eat rocks" or "put glue on pizza" due to retrieving satirical/erroneous webpages). This raises a fundamental, bidirectional question about how LLMs **arbitrate** between two potentially conflicting information sources:
- If retrieved content is **incorrect**, does the model correctly ignore it and rely on its own (correct) internal knowledge, or does it blindly recapitulate the retrieved error?
- Conversely, if the model's own internal (parametric) knowledge is **incorrect**, does it correctly defer to and adopt correct retrieved information, or does it stubbornly insist on its own wrong prior answer?

Prior work (Longpre et al. 2021; Xie et al. 2023) had begun exploring this "knowledge conflict" tension, but largely tested only one direction (context wrong, model right) — a setup where a model could trivially "succeed" simply by always ignoring context. ClashEval addresses this gap by constructing a dataset that includes **both** directions of conflict (context-right/model-wrong, and context-wrong/model-right), forcing genuine arbitration rather than a naive ignore-everything strategy, and additionally elicits a **quantitative relationship** between (1) the model's own confidence in its prior answer (via token log-probabilities) and (2) the magnitude/degree of deviation of the retrieved (perturbed) content from the ground truth.

**Proposed Methodology:**
The authors construct **ClashEval**, a benchmark of 1,294 fact-based questions spanning six domains (Drug Dosages, News, Wikipedia Dates, Sports Records, Names, Locations), each paired with a real reference document containing the answer. For each question:
1. The model is first asked the question **without context** to elicit its **prior response** $r(q)$ (using only parametric knowledge).
2. The reference document is then **systematically perturbed** — the correct answer value is replaced with a modified value at varying degrees of deviation (e.g., multiplicative factors of 0.1× to 10× for numerical datasets, or slight/significant/comical categorical substitutions for names/locations).
3. The model is asked the same question again, this time **with the (possibly perturbed) document as context**, eliciting a **contextual response** $r(q|c)$.
4. The authors compare $r(q)$, $r(q|c)$, the ground-truth answer, and (where available) the token log-probabilities of each response, to compute three core metrics: **Accuracy**, **Prior Bias**, and **Context Bias** (formally defined in Section 2 below).

Six top-performing LLMs are benchmarked: GPT-4o, GPT-3.5 (`gpt-3.5-turbo-0125`), Llama-3-8B-Instruct, Gemini 1.5 Flash, Claude Opus, and Claude Sonnet.

**Main Contributions:**
- Introduces **ClashEval**, an open-sourced QA benchmark of 1,200+ questions across six domains, each paired with a real contextual document whose embedded answer is systematically perturbed across a spectrum of erroneous values (from subtle to blatant).
- Benchmarks six top-performing LLMs on this dataset, reporting three formally-defined metrics: **Accuracy**, **Prior Bias**, and **Context Bias**.
- Finds that LLMs are highly susceptible to adopting **incorrect** retrieved content, overriding their own correct prior knowledge **over 60% of the time** (GPT-4o specifically).
- Establishes a **quantitative negative relationship** between (a) the degree/magnitude of deviation of the retrieved content from the truth and the model's context-preference rate, and (b) the model's own token-probability confidence in its prior response and its context-preference rate.
- Proposes and validates simple **token-probability-based correction methods** (raw and calibrated) that meaningfully improve accuracy and reduce context bias by comparing the model's confidence with vs. without context.
- Provides supplementary analyses on multi-document RAG (adding up to 4 additional retrieved documents) and on prompt-wording effects (strict vs. standard vs. loose instructions) on context adherence.

---

## 2. Detailed Recreation of All Tables

### Formal Metric Definitions (used throughout the tables)

Given a QA instance $x = (q, c)$ where $q$ is the query and $c$ is the (possibly perturbed) context, let $r(q)$ be the model's **prior response** (no context) and $r(q|c)$ be its **contextual response** (with context):

$$\text{Accuracy} = Pr\big(r(q|c) \text{ is right} \mid c \text{ is right or } r(q) \text{ is right}\big)$$

$$\text{Prior Bias} = Pr\big(r(q|c) \text{ is wrong} \mid c \text{ is right and } r(q) \text{ is wrong}\big)$$

$$\text{Context Bias} = Pr\big(r(q|c) \text{ is wrong} \mid c \text{ is wrong and } r(q) \text{ is right}\big)$$

---

### Table 1: Statistics for each dataset

| Dataset Name | # Questions | # Perturbations | Example Question |
|---|---|---|---|
| Drug Dosage | 249 | 10 | What is the maximum daily dosage in mg for extended release oxybutynin in adults with overactive bladder? |
| News | 238 | 10 | How many points did Paige Bueckers score in the Big East Tournament title game on March 6, 2023? |
| Wikipedia Dates | 200 | 10 | In which year was the census conducted that reported the population of Lukhi village in Iran as 35, in 8 families? |
| Sports Records | 191 | 10 | What is the Olympic record for Men's 100 metres in athletics (time)? |
| Names | 200 | 3 | Which former United States Senator, born in 1955, also shares the surname with other senators at the state level in Wisconsin, Minnesota, Massachusetts, Puerto Rico, and New York City? |
| Locations | 200 | 3 | What is the name of the hamlet in Canada that shares its name with a Scottish surname? |

**Key Takeaway:** Defines the composition of the 1,294-question ClashEval benchmark, spanning both numerically-answered domains (Drug Dosage, News, Dates, Records — each with 10 graded perturbation levels) and categorically-answered domains (Names, Locations — each with 3 qualitative perturbation levels: slight, significant, comical). News is specifically included as an out-of-distribution domain that cannot be answered correctly from parametric knowledge alone.

---

### Table 2: Model behavior given a subset of the data where either the prior or the context is correct

| Model | Response | Prior Correct | Context Correct |
|---|---|---|---|
| **Claude Opus** | Prior | 0.585 (0.550, 0.619) | 0.042 (0.027, 0.058) |
| | Context | 0.313 (0.282, 0.346) | 0.901 (0.879, 0.923) |
| | Neither | 0.102 (0.082, 0.125) | 0.057 (0.040, 0.075) |
| **Claude Sonnet** | Prior | 0.436 (0.403, 0.469) | 0.051 (0.037, 0.067) |
| | Context | 0.401 (0.374, 0.434) | 0.881 (0.859, 0.903) |
| | Neither | 0.163 (0.138, 0.186) | 0.068 (0.052, 0.086) |
| **Gemini 1.5** | Prior | 0.388 (0.362, 0.416) | 0.074 (0.058, 0.091) |
| | Context | 0.490 (0.461, 0.521) | 0.860 (0.838, 0.881) |
| | Neither | 0.122 (0.103, 0.143) | 0.066 (0.051, 0.082) |
| **GPT-4o** | Prior | 0.327 (0.293, 0.358) | 0.041 (0.027, 0.056) |
| | Context | 0.608 (0.571, 0.643) | 0.903 (0.881, 0.923) |
| | Neither | 0.065 (0.047, 0.083) | 0.056 (0.040, 0.072) |
| **GPT-3.5** | Prior | 0.237 (0.213, 0.263) | 0.057 (0.043, 0.072) |
| | Context | 0.626 (0.598, 0.657) | 0.841 (0.817, 0.865) |
| | Neither | 0.137 (0.113, 0.160) | 0.102 (0.082, 0.123) |
| **Llama-3** | Prior | 0.208 (0.185, 0.230) | 0.041 (0.029, 0.054) |
| | Context | 0.529 (0.499, 0.558) | 0.793 (0.767, 0.818) |
| | Neither | 0.263 (0.236, 0.291) | 0.166 (0.145, 0.191) |

*Values shown as point estimate with 95% confidence interval in parentheses. "Prior Correct" columns report response rates when only the prior answer is correct (context is wrong); "Context Correct" columns report response rates when only the context answer is correct (prior is wrong).*

**Key Takeaway:** GPT-4o and GPT-3.5 choose "Context" over 60% of the time even when the context is wrong and the prior is right (0.608 and 0.626 respectively) — directly illustrating strong context bias. Claude Opus is the most resistant to incorrect context (only 0.313 "Context" rate when context is wrong), while all models overwhelmingly select "Context" (~0.79–0.90) when the context happens to be correct, showing reasonable ability to adopt correct external information. Llama-3 and GPT-3.5 show the highest "Neither" rates, indicating a tendency to produce answers matching neither source when confused.

---

### Table 3: Accuracy, context bias, and prior bias under three correction conditions

| Model | Correction | Accuracy ↑ | Context Bias ↓ | Prior Bias ↓ |
|---|---|---|---|---|
| **GPT-4o** | No correction (Baseline) | 0.615 (0.595, 0.636) | 0.304 (0.287, 0.321) | 0.021 (0.014, 0.028) |
| | Token Probability Correction | 0.693 (0.672, 0.714) | 0.194 (0.177, 0.210) | 0.043 (0.032, 0.053) |
| | Calibrated Token Prob. Correction | 0.754 (0.733, 0.775) | 0.107 (0.093, 0.122) | 0.085 (0.072, 0.098) |
| **GPT-3.5** | No correction (Baseline) | 0.539 (0.521, 0.557) | 0.313 (0.298, 0.328) | 0.028 (0.021, 0.036) |
| | Token Probability Correction | 0.596 (0.575, 0.616) | 0.253 (0.237, 0.269) | 0.056 (0.046, 0.067) |
| | Calibrated Token Prob. Correction | 0.701 (0.678, 0.722) | 0.110 (0.098, 0.124) | 0.147 (0.132, 0.164) |
| **Llama-3** | No correction (Baseline) | 0.500 (0.483, 0.515) | 0.264 (0.250, 0.279) | 0.021 (0.015, 0.027) |
| | Token Probability Correction | 0.556 (0.537, 0.574) | 0.235 (0.220, 0.249) | 0.046 (0.037, 0.055) |
| | Calibrated Token Prob. Correction | 0.649 (0.627, 0.669) | 0.111 (0.099, 0.122) | 0.188 (0.173, 0.204) |

**Key Takeaway:** Both correction methods (comparing token probabilities of the prior vs. contextual response) improve accuracy and reduce context bias for all three models that expose token probabilities. **Calibrated** Token Probability Correction (comparing probability *percentiles* rather than raw scores, since raw context-response probabilities are right-skewed while prior probabilities are near-uniform) yields the largest gains — e.g., GPT-4o's accuracy rises from 0.615 to 0.754 and context bias drops from 0.304 to 0.107 — but at the cost of a substantial increase in prior bias (e.g., GPT-4o's prior bias rises from 0.021 to 0.085; Llama-3's from 0.021 to 0.188). The text notes this calibrated method still outperforms a naive random-replacement baseline at matched bias rates (75.4% vs. 57.5% accuracy at 8.5% prior bias for GPT-4o).

---

### Table 4 (Appendix): Comparison of six top-performing models across three metrics

| Model | Context Bias ↓ | Prior Bias ↓ | Accuracy ↑ |
|---|---|---|---|
| *Claude Opus* | **0.157** (0.141, 0.174) | **0.021** (0.014, 0.029) | **0.743** (0.723, 0.763) |
| *Claude Sonnet* | 0.201 (0.184, 0.215) | 0.025 (0.018, 0.033) | 0.658 (0.641, 0.678) |
| *Gemini 1.5* | 0.245 (0.231, 0.260) | 0.037 (0.029, 0.046) | 0.624 (0.607, 0.641) |
| *GPT-4o* | 0.304 (0.287, 0.321) | 0.021 (0.013, 0.028) | 0.615 (0.594, 0.633) |
| *GPT-3.5* | 0.313 (0.298, 0.329) | 0.028 (0.021, 0.036) | 0.539 (0.522, 0.558) |
| *Llama-3* | 0.264 (0.250, 0.280) | 0.021 (0.015, 0.027) | 0.500 (0.482, 0.518) |

*Bold = best value per column, as explicitly noted in the caption ("Claude Opus performs the best across all metrics with a context bias rate of 0.157").*

**Key Takeaway:** This is the aggregate, headline comparison table across all six benchmarked models. Claude Opus is the top performer on all three metrics simultaneously (highest accuracy, lowest context bias, tied-lowest prior bias), while Llama-3 has the lowest accuracy (0.500 — near random on the discernment task) despite a relatively moderate context bias, largely because it frequently answers with "neither" prior nor context (as seen in Table 2). Notably, GPT-4o — despite being a leading general-purpose model — has the second-worst context bias (0.304), second only to GPT-3.5, highlighting that general benchmark performance does not predict RAG-conflict robustness.

---

### Table 5 (Appendix): Accuracy and Mean Prior Probability Comparison Across Models and Datasets

**Claude Opus**

| Dataset | Acc. Without Context | Acc. With Correct Context |
|---|---|---|
| Drugs | 0.566 | 0.827 |
| Locations | 0.550 | 0.935 |
| Names | 0.400 | 0.995 |
| News | 0.109 | 0.966 |
| Records | 0.717 | 0.953 |
| Years | 0.490 | 0.980 |

**Claude Sonnet**

| Dataset | Acc. Without Context | Acc. With Correct Context |
|---|---|---|
| Drugs | 0.534 | 0.775 |
| Locations | 0.405 | 0.930 |
| Names | 0.285 | 0.995 |
| News | 0.0966 | 0.937 |
| Records | 0.508 | 0.880 |
| Years | 0.215 | 0.980 |

**Gemini 1.5 Flash**

| Dataset | Acc. Without Context | Acc. With Correct Context |
|---|---|---|
| Drugs | 0.213 | 0.735 |
| Locations | 0.325 | 0.920 |
| Names | 0.200 | 0.995 |
| News | 0.0840 | 0.958 |
| Records | 0.508 | 0.843 |
| Years | 0.205 | 0.990 |

**GPT-4o**

| Dataset | Acc. Without Context | Acc. With Correct Context | Mean Prior Prob |
|---|---|---|---|
| Drugs | 0.578 | 0.863 | 0.818 |
| Locations | 0.575 | 0.925 | 0.877 |
| Names | 0.445 | 0.990 | 0.847 |
| News | 0.0882 | 0.971 | 0.469 |
| Records | 0.628 | 0.921 | 0.498 |
| Years | 0.540 | 0.990 | 0.773 |
| **All** | **0.467** | **0.941** | **0.675** |

**GPT-3.5**

| Dataset | Acc. Without Context | Acc. With Correct Context | Mean Prior Prob |
|---|---|---|---|
| Drugs | 0.446 | 0.751 | 0.727 |
| Locations | 0.410 | 0.875 | 0.838 |
| Names | 0.295 | 0.985 | 0.819 |
| News | 0.0630 | 0.908 | 0.232 |
| Records | 0.592 | 0.796 | 0.578 |
| Years | 0.295 | 0.980 | 0.596 |
| **All** | **0.344** | **0.879** | **0.573** |

**Llama 3**

| Dataset | Acc. Without Context | Acc. With Correct Context | Mean Prior Prob |
|---|---|---|---|
| Drugs | 0.317 | 0.598 | 0.793 |
| Locations | 0.290 | 0.915 | 0.853 |
| Names | 0.165 | 0.925 | 0.770 |
| News | 0.0714 | 0.912 | 0.608 |
| Records | 0.377 | 0.524 | 0.757 |
| Years | 0.160 | 0.975 | 0.720 |
| **All** | **0.228** | **0.805** | **0.732** |

**Key Takeaway:** Across every model and dataset, accuracy with correct context is dramatically higher than accuracy without context — most strikingly for **News** (e.g., GPT-4o: 0.088 → 0.971; Claude Opus: 0.109 → 0.966), confirming News questions are genuinely out-of-distribution/unanswerable from parametric knowledge alone, so RAG provides enormous value there. Models also show a very low mean prior probability specifically on News (e.g., GPT-4o: 0.469, GPT-3.5: 0.232) relative to other domains, consistent with their low confidence and low without-context accuracy on this domain. Records and Drugs show the smallest context-free-to-context-given accuracy gap for some models, suggesting relatively stronger baseline parametric knowledge there.

---

### Table 6 (Appendix): Accuracy comparison of GPT-4o and Claude Opus with 1 vs. 5 retrieved documents

**GPT-4o**

| Dataset | Acc. Without Context | Acc. With Correct Context (k=1) | Acc. With Correct Context (k=5) |
|---|---|---|---|
| Drugs | 0.578 | 0.863 | 0.819 |
| Locations | 0.575 | 0.925 | 0.925 |
| Names | 0.445 | 0.990 | 0.985 |
| News | 0.088 | 0.971 | 0.924 |
| Records | 0.628 | 0.921 | 0.911 |
| Years | 0.540 | 0.990 | 0.990 |
| **All** | **0.467** | **0.941** | **0.922** |

**Claude Opus**

| Dataset | Acc. Without Context | Acc. With Correct Context (k=1) | Acc. With Correct Context (k=5) |
|---|---|---|---|
| Drugs | 0.566 | 0.827 | 0.719 |
| Locations | 0.550 | 0.935 | 0.875 |
| Names | 0.400 | 0.995 | 0.880 |
| News | 0.109 | 0.966 | 0.853 |
| Records | 0.717 | 0.953 | 0.822 |
| Years | 0.490 | 0.980 | 0.935 |
| **All** | **0.463** | **0.939** | **0.843** |

**Key Takeaway:** Adding four additional (distractor) retrieved documents (k=5 total vs. k=1) consistently **lowers** accuracy for both GPT-4o and Claude Opus relative to using only the single correct document, across nearly every dataset — most notably for Claude Opus (overall accuracy drops from 0.939 to 0.843, a larger degradation than GPT-4o's drop from 0.941 to 0.922). This supports the paper's claim that longer/multi-document contexts can dilute a model's ability to correctly leverage the relevant information, consistent with prior findings that LLMs perform worse on longer contexts.

---

### Table 7 (Appendix): Comparison of prior and context choices between Claude Opus and GPT-4o for k=1 and k=5 documents

**Claude Opus, k=1**

| Response | Prior Correct | Context Correct |
|---|---|---|
| Prior Chosen | 0.608 (0.575, 0.646) | 0.042 (0.028, 0.058) |
| Context Chosen | 0.287 (0.255, 0.318) | 0.901 (0.878, 0.923) |
| Neither Chosen | 0.105 (0.082, 0.129) | 0.057 (0.039, 0.074) |

**Claude Opus, k=5**

| Response | Prior Correct | Context Correct |
|---|---|---|
| Prior Chosen | 0.618 (0.584, 0.652) | 0.067 (0.050, 0.085) |
| Context Chosen | 0.237 (0.209, 0.267) | 0.778 (0.747, 0.810) |
| Neither Chosen | 0.145 (0.121, 0.172) | 0.155 (0.130, 0.181) |

**GPT-4o, k=1**

| Response | Prior Correct | Context Correct |
|---|---|---|
| Prior Chosen | 0.355 (0.321, 0.388) | 0.041 (0.027, 0.057) |
| Context Chosen | 0.582 (0.549, 0.617) | 0.903 (0.881, 0.925) |
| Neither Chosen | 0.064 (0.048, 0.081) | 0.056 (0.039, 0.074) |

**GPT-4o, k=5**

| Response | Prior Correct | Context Correct |
|---|---|---|
| Prior Chosen | 0.535 (0.498, 0.569) | 0.044 (0.029, 0.060) |
| Context Chosen | 0.383 (0.349, 0.416) | 0.868 (0.843, 0.894) |
| Neither Chosen | 0.082 (0.061, 0.102) | 0.088 (0.069, 0.111) |

**Key Takeaway:** With 5 documents instead of 1, both models shift markedly toward choosing "Prior" more often (Claude Opus's Prior-Chosen-when-Prior-Correct rises slightly from 0.608 to 0.618; GPT-4o's rises substantially from 0.355 to 0.535) and "Neither" more often — while "Context Chosen" rates drop in both the Prior-Correct and Context-Correct columns. This confirms the paper's claim (Section 4.2) that multi-document RAG reduces context bias (models adhere to context less overall) but at the cost of increased "neither" responses and, ultimately, lower overall accuracy (as seen in Table 6) — a genuine trade-off rather than a strict improvement.

---

## 3. Figure Catalog & In-Depth Explanation

### Figure 1: Schematic of generating modified documents
**Description & Insights:** A flowchart using a concrete worked example: the question "What is the maximum daily dosage of olanzapine for the treatment of agitation/aggression associated with psychiatric disorders in adults?" is first posed to the LLM **without** any document, yielding a **Prior Response** of "20 mg." A **Reference Document** excerpt (stating "...maximum: 30 mg/day...") is then perturbed at three example multiplier levels — 0.1× ("...maximum: 3 mg/day..."), 2× ("...maximum: 60 mg/day..."), and 10× ("...maximum: 300 mg/day...") — producing three **Modified Documents**. Each modified document is fed back to the LLM as context, producing a **Response w/ Modified Context** (20 mg, 60.0 mg, and 20 mg respectively for the three perturbation levels shown) and a corresponding "LLM prefers: Prior / Context / Prior" judgment. This figure is the visual definition of the paper's core experimental pipeline: question → prior response (no context) → document perturbation → contextual response (with perturbed context) → comparison of which source ("prior" vs. "context") the final response matches.

---

### Figure 2: Examples from three datasets demonstrating differential LLM responses (GPT-4o) across various types of context modifications
**Description & Insights:** A table-style figure showing five worked examples across five datasets (Drug Dosages, Sports Records, Dates, Names, Locations), each listing: the example question, the ground-truth Answer, the model's Response without Context, the Modification type/magnitude, the resulting Value in the (perturbed) document, the model's Response with Context, and a checkmark/X indicating whether the model "Preferred Context." Responses shown in red are wrong (differ from the true answer); responses in green are correct. For example, in the Drug Dosages row, at a 0.1× perturbation (value 3 in document) the model still answers "20" (its own prior, correctly resisting the bad context, marked X = did not prefer context); but at 0.4× (value 12) it also answers "20" (still resisting); at Reference (unperturbed, value 30) it correctly answers "30" (adopting the correct context, marked ✓); at 1.5× (value 45) it again adopts context answering "45" (✓, but this is now a **wrong** answer since it's a perturbed/incorrect document — shown in red); and at 10× (value 300) it reverts to its prior "20" (X). This figure concretely illustrates the paper's central "tug-of-war" phenomenon at the level of individual model responses, and shows non-monotonic behavior in the Drug Dosages example (adopting context at 1.5x but not at 0.4x or 10x), while other examples (Sports Records, Dates) show clearer patterns of context adoption at small perturbations and increasing prior-reliance at larger deviations.

---

### Figure 3: Context preference rate vs. degree of context modification (six domain panels)
**Description & Insights:** A 2×3 grid of scatter plots with best-fit trendlines, one panel per domain (Drugs, News, Years, Records, Names, Locations). Y-axis is "Context preferred (%)" (or "RAG Preference Rate (%)" for the categorical Names/Locations panels); x-axis is "Absolute Log Fold Change" (for the four numerical datasets, up to two log-fold changes) or "Absolute Value Change" (for Years, in raw year units) or "Modification Type" (for Names/Locations: none, slight, significant, comical). Six colored lines/point-sets represent the six benchmarked models (GPT-4o, Llama-3, GPT-3.5, Claude Opus, Claude Sonnet, Gemini 1.5). Across all six panels, context preference rate **decreases** as the magnitude of the modification/deviation increases — i.e., the more obviously wrong the perturbed content is, the less likely models are to adopt it. Notably, models differ in **both intercept and slope**: Claude Opus (red line) consistently sits at or near the bottom of each panel with a comparatively steep negative slope, indicating both lower baseline context adoption and faster rejection as errors grow more blatant, whereas other models (e.g., GPT-4o, blue) sit higher with shallower slopes. The text highlights that Claude Opus adheres to incorrect content ~30% less than GPT-4o for the same magnitude of perturbation.

---

### Figure 4: Context preference rate vs. prior token probability (six domain panels, three models)
**Description & Insights:** Another 2×3 grid of scatter plots with best-fit trendlines and slopes labeled in the legend (only the three models that expose token probabilities are shown: GPT-4o, Llama-3, GPT-3.5). Y-axis is "Context Preferred (%)"; x-axis is "Prior Probability" (0.2–1.0, binned into 10 equidistant bins across [0.0, 1.0] for visualization). Across all six domains, there is a consistent **negative** relationship: the more confident the model is in its own prior answer (higher token probability), the **less** likely it is to prefer the retrieved context. Slopes vary substantially by domain and model — ranging from roughly -0.11 (GPT-3.5 on Drugs) to as steep as -1.67 (GPT-3.5 on Records) and -0.92 (GPT-3.5 on Names) — indicating that some domains (e.g., Dates) make models highly susceptible to context regardless of confidence, while others (e.g., News, where prior confidence is generally low anyway) show more consistently high context adoption. A slope of, e.g., -0.45 is interpreted as a 4.5-percentage-point drop in context-preference likelihood per 10-percentage-point increase in prior-response probability. This figure directly motivates the token-probability-based correction methods presented in Table 3.

---

### Figure 5 (Appendix): Bar chart of Model Accuracy and Biases (from Table 4)
**Description & Insights:** A grouped bar chart with three metric groups on the x-axis (Accuracy, Context Bias, Prior Bias) and "Rate" (0.0–0.8) on the y-axis, with six colored bars per group (one per model: Claude Opus, Claude Sonnet, Gemini 1.5 Flash, GPT-4o, GPT-3.5, Llama-3-8b-Instruct), each with error bars representing 95% confidence intervals. This is a direct visual re-presentation of Table 4's numbers: Claude Opus (blue) has the tallest bar in Accuracy (~0.74) and among the shortest in Context Bias (~0.16); GPT-3.5 and Llama-3 have the shortest Accuracy bars (~0.54, ~0.50); Prior Bias bars are uniformly small (~0.02–0.04) across all models, showing that prior bias is a comparatively minor issue relative to context bias for all six models tested. The figure makes the cross-model ranking pattern from Table 4 immediately visually apparent.

---

### Figure 6 (Appendix): Effect of different prompts on context preference rate vs. prior probability
**Description & Insights:** Two scatter plots with best-fit trendlines (Drug Dosages and Wikipedia Names panels), each showing "RAG Preference Rate" (y-axis) vs. "Prior Probability" (x-axis) for GPT-4 under three different prompt-wording conditions, each with its own colored trendline and labeled slope: **Strict** prompt ("You MUST absolutely strictly adhere to the following piece of context in your answer. Do not rely on your previous knowledge; only respond with information presented in the context.") — red line, slopes of -0.27 (Drugs) and -0.06 (Names); **Standard** prompt ("Use the following pieces of retrieved context to answer the question.") — gray line, slopes of -0.26 (Drugs) and -0.15 (Names); **Loose** prompt ("Consider the following piece of retrieved context to answer the question, but use your reasonable judgment based on what you know about <subject>.") — blue line, slopes of -0.41 (Drugs) and -0.18 (Names). The Loose prompt produces both a lower overall context-preference rate and a steeper (more negative) slope relative to prior confidence, compared to the Strict prompt, which keeps context preference nearly flat and high (especially in the Names panel, staying near 1.0 regardless of prior confidence). This demonstrates that prompt wording is itself a significant lever for controlling how much a model defers to retrieved context — independent of the model's underlying confidence-based tendencies documented in Figure 4.

---

## 4. Existing Limitations

The authors explicitly enumerate the following limitations in the **Discussion** section:

1. **Limited domain coverage:** RAG systems are deployed across far more domains than the six covered in this benchmark (Drug Dosages, News, Wikipedia Dates, Sports Records, Names, Locations); findings may not generalize to all real-world RAG use cases.
2. **Simplicity of question generation:** To keep experiments tractable, the question-generation process is strictly fact-based (single, extractable numerical or short-answer facts) and does not require multi-step logic, document synthesis, or other higher-level reasoning — so the benchmark does not test more complex RAG scenarios involving reasoning across multiple facts or documents.
3. **Enriched error rate, not representative of real-world bias rates:** The dataset intentionally contains an artificially high (enriched) rate of contextual errors via systematic perturbation, so the reported bias/accuracy metrics are **not** meant to represent the actual frequency of such conflicts "in the wild" (i.e., they measure model behavior *conditional on* a conflict existing, not the base rate of conflicts).
4. **Token-probability method applicability:** The proposed token-probability correction method (Table 3) only applies to models that expose token/log probability outputs, limiting its applicability to closed models without such access (e.g., only GPT-4o, GPT-3.5, and Llama-3 could be evaluated with this method among the six benchmarked models).
5. **Dual-use / misuse risk:** Although the dataset is intended to help improve LLMs' ability to provide accurate information, the authors acknowledge that bad actors could use the documented per-model shortcomings (e.g., knowing exactly how much perturbation is needed to flip a specific model's answer) to deliberately exploit or manipulate RAG-based systems.

**Additional technical constraints and generalizability bottlenecks identified elsewhere in the paper:**
- **Multi-document trade-off:** Adding more retrieved documents (k=5 vs. k=1) reduces context bias but also reduces overall accuracy and increases "neither" (unclear/hedged) responses, per Tables 6–7 — indicating no simple fix without trade-offs.
- **Calibration trade-off:** The best-performing correction method (Calibrated Token Probability Correction) substantially reduces context bias but at the cost of a large increase in prior bias (e.g., Llama-3's prior bias rises nearly 9-fold, from 0.021 to 0.188), showing that current correction methods shift rather than eliminate the underlying tension.
- **Prompt sensitivity:** Figure 6 shows that even holding the model and dataset constant, prompt wording (strict/standard/loose) meaningfully changes context-adherence behavior — meaning reported bias rates are partly an artifact of the specific "standard" RAG prompt template chosen (based on popular open-source libraries LangChain/LlamaIndex) rather than a fixed, prompt-independent model property.
- **No formal compute-cost/scaling discussion:** Per the NeurIPS Paper Checklist included in the paper, the authors explicitly note they do not report their own compute resources ("We do not use our own compute – models are run on commercial APIs"), and no experimental training/hyperparameter details are relevant since no models were trained (all evaluation is via API inference on existing models).
- **Single-document assumption for main experiments:** The paper's primary analysis (Sections 4.1, 4.3, 4.4) focuses on the single-document (k=1) RAG setting; multi-document analysis (Section 4.2) is a secondary, more limited exploration covering only two of the six models (GPT-4o and Claude Opus).

---

## 5. Conclusion & Future Outlook

**High-Level Conclusion:** ClashEval provides the first systematic, bidirectional benchmark for studying how LLMs arbitrate between their own internal (parametric) knowledge and external evidence provided via RAG when the two are in conflict. Across six top-performing LLMs (GPT-4o, GPT-3.5, Llama-3-8B-Instruct, Gemini 1.5 Flash, Claude Opus, Claude Sonnet) and six knowledge domains, the central finding is that even the most capable models exhibit **strong context bias**, overriding their own correct prior knowledge more than 60% of the time when presented with incorrect retrieved content (most pronounced in GPT-4o and GPT-3.5). This bias is not absolute, however — models are progressively less likely to adopt retrieved content as its deviation from the truth becomes more blatant, and each model has a distinct "prior distribution over truthfulness" that varies by domain (e.g., Claude Opus is markedly more resistant to bad context than GPT-4o at matched perturbation magnitudes). The paper also establishes that models are reasonably well **self-calibrated** in a directional sense — lower confidence (token probability) in their own prior answer correlates with higher likelihood of deferring to context — and shows that this signal can be exploited via simple token-probability-based correction methods to meaningfully improve accuracy and reduce context bias (though at the cost of increased prior bias). Notably, the paper highlights that strong general-purpose benchmark performance (e.g., GPT-4o's leading position on LMSYS Chatbot Arena as of the paper's writing) does **not** predict robustness in RAG-conflict settings, where smaller/different models (e.g., Claude Sonnet, Claude Opus) can outperform it.

**Future Work / Extensions (explicitly suggested or implied by authors):**
- Further investigation into probability calibration methods as a "promising approach" for reducing both prior and context bias — the authors explicitly frame their proposed correction methods as a **baseline for future methods** rather than a final solution.
- Extension of the benchmark to additional domains beyond the six covered, and to more complex, multi-step reasoning or multi-document-synthesis question types (acknowledged as an intentional simplification in the current work).
- Deeper exploration of multi-document RAG dynamics, given the accuracy/context-bias trade-off observed when scaling from k=1 to k=5 documents.
- Development of RAG systems with more explicit, controllable, and predictable policies for how and when to defer to external evidence versus internal knowledge — the authors note that "the lack of explicit expectations around how models will decide to use contextual information remains a risk."

**Practical Implications:** As retrieval-augmented AI systems become increasingly prevalent in high-stakes domains (the paper specifically highlights medical/drug-dosage information as a use case), ClashEval provides both a diagnostic benchmark and early evidence that naive RAG deployment carries real risk of confidently propagating retrieval errors — with direct real-world resonance to documented incidents like Google's AI Overview recommending harmful advice due to erroneous retrieved content. The authors frame resolving this prior-vs-context tension as "a crucial challenge on the path to safe and trustworthy language models," and their open-sourced dataset and evaluation code are intended to support continued benchmarking of future top-performing models on this specific capability.

---

*Note: This paper contains numerous citations to related and prior work (e.g., Longpre et al. 2021 and Xie et al. 2023 for knowledge-conflict foundations, Lewis et al. 2020 for RAG, various RAG-evaluation-framework papers such as RAGAS, ARES, and RetrievalQA, and news/media citations regarding real-world AI Overview incidents) which have been transcribed directly from the source PDF's reference list and are not independently verified or hallucinated by this summarization process. The paper also includes a full NeurIPS "Paper Checklist" appendix (documenting compliance with claims, limitations, reproducibility, ethics, and licensing guidelines), which has been reflected above only insofar as it contains substantive content (e.g., the note that no proprietary compute was used, as all models were accessed via commercial APIs).*
