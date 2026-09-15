# Disentangling Parametric Priors from Contextual Reliance in MCQ Evaluation

**A knowledge-dependency metric for Automatic Question Generation that survives modern LLMs.**

This repository contains the code, data pipeline, and experimental results for a research
project that asks a single question: *when an automatic evaluator says a generated
multiple-choice question "requires the learning material," is that claim actually true?*

We show — empirically, on 884 SciQ questions — that it frequently is not. A large share of
the "the fact made this answerable" signal that the KDA metric (Moon et al., EMNLP 2022)
attributes to the target material is in fact produced by the solver's **parametric prior**,
not by its reading of the material. We then propose and implement a **counterfactual
context perturbation** procedure that separates the two, and report a corrected metric.

---

## Table of Contents

1. [Project Overview & Research Story](#1-project-overview--research-story)
2. [Theoretical Foundations & KDA Formulations](#2-theoretical-foundations--kda-formulations)
3. [Related Works & Key Inspirations](#3-related-works--key-inspirations)
4. [Repository Structure & Directory Map](#4-repository-structure--directory-map)
5. [Quickstart & Execution Guide](#5-quickstart--execution-guide)
6. [Findings Pipeline & Current Results](#6-findings-pipeline--current-results)

---

## 1. Project Overview & Research Story

### 1.1. Paper Framing

This project follows the **Finding Paper** paradigm:

$$\textbf{Finding} \;\longrightarrow\; \textbf{Evaluation Method} \;\longrightarrow\; \textbf{Application / Validation}$$

| Stage | Content | Status |
|---|---|---|
| **Finding** | Existing knowledge-dependency evaluators conflate *contextual grounding* with *parametric recall*. The conflation is large enough to change the metric's ranking of questions. | ✅ Measured (Experiment 2) |
| **Evaluation Method** | A counterfactual-adjusted knowledge-dependency metric that credits a question only when the solver's correctness is demonstrably caused by the target material. | ✅ Implemented and run |
| **Application / Validation** | Use the metric to filter and rerank generated quizzes; validate against teacher/human judgement. | 🔜 Planned (Experiment 3) |

### 1.2. Motivation & Background

Automatic Question Generation (AQG) for education has an evaluation problem that n-gram
metrics cannot touch. BLEU, ROUGE and METEOR ask *"does this question resemble a reference
question?"* — but the property that actually matters pedagogically is different:

> A multiple-choice question designed to test a target fact must be answerable **only** by a
> student who knows that fact, and **not** by a student who does not.

KDA (Knowledge Dependent Answerability) formalized exactly this and estimated it by running
an ensemble of pre-trained language models as *simulated students*, comparing their answer
correctness with and without the target fact in the prompt. The construction rests on one
load-bearing assumption:

> **The solver does not already know the answer.**

That assumption was defensible for a 355M-parameter RACE-tuned encoder in 2022. It is not
defensible for a modern LLM.

### 1.3. Core Research Questions

#### RQ1 — Finding: a failure mode of modern LLM evaluators

> **Do modern LLM-based evaluators reliably measure whether a question genuinely requires
> the target learning material?**

Consider a course note reading *"ResNet introduces residual connections to alleviate
degradation in deep networks,"* and a generated question:

> *"What architecture introduced residual connections?"* → (a) VGG (b) ResNet (c) LSTM (d) BERT

A modern LLM answers **(b)** correctly with the passage removed. Its parametric memory
already contains the fact. Under the KDA construction, the "without fact" probability
$P(R^q = 0)$ collapses toward zero, the question's weight in the metric vanishes, and the
evaluator can no longer tell a genuinely knowledge-dependent question from a trivial one.
Worse, in the aggregate the metric *credits* the question: the solver is correct with the
fact present, and the machinery has no way to check *why*.

The original KDA paper anticipated part of this. It documents **"too easy questions"**
(§6.2) and the **"difficulty of measuring PLM ignorance"** (§6.3) — the authors could not
inspect their solvers' pretraining corpora and therefore could not guarantee the
without-fact condition was a true ignorance baseline. They explicitly flagged as future
work the need for *"language models with reading comprehension ability but without prior
factual knowledge."*

With large LLMs that problem is no longer a caveat — it is the dominant term. And as
**ClashEval** (NeurIPS 2024) demonstrated, the prior-versus-context arbitration is not a
simple threshold: models blend the two sources in a confidence-dependent, domain-dependent,
even prompt-wording-dependent way. Disentangling parametric priors from contextual reliance
is therefore a non-trivial measurement problem, not a data-cleaning step.

**What we measure.** Instead of assuming ignorance, we *test* for it, in two ways.
Section 6.3 reports that on SciQ, between 8% and 64% of the questions a 355M-parameter
solver gets right with the passage are questions it would still get right if the passage
said something else entirely. Section 6.4 then runs the metric with a modern instruct LLM
and finds the sharper version of the same problem: **Qwen3-4B answers 95.4% of SciQ
correctly with no passage at all**, which zeroes those questions' weight and leaves
$KDA_{disc}$ computed from 41 of 884 questions — 4.6% of the dataset.

#### RQ2 — Evaluation Method

> **How can we formulate a robust metric that measures whether an MCQ answer strictly
> depends on the provided target knowledge material, versus prior knowledge or surface
> shortcuts?**

Our answer: add a **third setting** in which the passage is minimally rewritten to assert a
*distractor* instead of the gold answer, and require the solver to *follow the context* in
order to earn credit. A solver that keeps answering the gold answer when the context has
been flipped was never reading the context. This yields three adjusted estimators
(hard / soft / sample-exclusion) defined in [§2.4](#24-the-theoretical-gap).

#### RQ3 — Application / Downstream Validation

> **Does the metric improve the quizzes that reach students?**

Apply the metric as a filter and reranker over generated quiz banks, comparing three arms —
**No Filtering** vs. **KDA Filtering** vs. **Our Disentangled Filtering** — and validate
with human/teacher evaluation that students genuinely need the course materials to answer
the surviving questions. This is the planned Experiment 3; the methodology draws directly on
**QG-SMS** (ACL 2025) for item analysis and student simulation.

---

## 2. Theoretical Foundations & KDA Formulations

### 2.1. Human ground-truth KDA

KDA is the probability that a student who answered *incorrectly* without the target fact
answers *correctly* once the fact is supplied. Let $r_j^q \in \\{0,1\\}$ be student $j$'s
correctness on question $q$ alone, and $r_j^{q+f}$ their correctness on $q$ with fact $f$:

$$KDA(q) = P\left(R^{q+f} = 1 \mid R^{q} = 0\right) = \frac{\sum_{j} \left(1 - r_j^{q}\right) r_j^{q+f}}{\sum_{j} \left(1 - r_j^{q}\right)}$$

The denominator restricts the estimate to students who did **not** already know the answer.
This conditioning is the entire point of the metric — and, as we argue, its weak point.

Measuring this directly requires a human trial (the original paper ran 116 students over
480 MCQs), so two PLM-based surrogates replace the students $j$ with language models $k$.

### 2.2. Discrete PLM approximation ($KDA_{disc}$)

Replace human binary correctness with PLM binary correctness — whether the gold option
carries the maximum logit, with and without the fact prepended:

$$KDA_{disc}(q) = \frac{\sum_{k} \left(1 - r_k^{q}\right) r_k^{q+f}}{\sum_{k} \left(1 - r_k^{q}\right)}$$

### 2.3. Continuous PLM approximation ($KDA_{cont}$)

Use the softmax probability mass on the gold option rather than the binarized argmax. This
is smoother and avoids the degenerate case where every solver answers correctly without the
fact and the discrete denominator becomes zero:

$$KDA_{cont}(q) = \frac{\sum_{k} P\left(R_k^{q} = 0\right) \, P\left(R_k^{q+f} = 1\right)}{\sum_{k} P\left(R_k^{q} = 0\right)}$$

The weight $P(R_k^q = 0) = 1 - P(R_k^q = 1)$ is the solver's *ignorance mass*. It is exactly
what our work challenges.

> **Note on ensemble size.** With $|M| = 1$ the weight $\left(1 - P(R^q{=}1)\right)$ appears
> in both numerator and denominator and cancels exactly, so a single model's $KDA_{cont}$
> degenerates to $P(R^{q+f}{=}1)$. A non-degenerate score requires $|M| \geq 2$; this
> repository enforces that unless `--allow-single-model` is passed.

### 2.4. The theoretical gap

$KDA_{cont}$ treats $P(R^q = 0)$ as a proxy for *"the student does not know the fact."* Two
distinct causes are collapsed into that one number:

| Cause of $P(R^{q+f}{=}1) > P(R^{q}{=}1)$ | What the metric should do | What the metric actually does |
|---|---|---|
| The solver **read** the fact in the passage | Credit the question | Credits it ✅ |
| The solver **already knew** the fact and the passage merely raised its confidence | Discount the question | Credits it ❌ |

Formally, the estimator assumes

$$P\left(R^{q+f}=1 \mid R^{q}=0\right) \;\approx\; P\left(\text{answer is caused by } f\right),$$

but the left-hand side is a *correlation* under a distribution shift, not a causal
quantity. It is only a valid causal estimate when the ignorance condition $R^q = 0$ truly
holds. Two failure modes break it:

1. **Parametric saturation.** As solver capacity grows, $P(R^q = 1) \to 1$ for factual
   questions, so $P(R^q = 0) \to 0$. Every question's weight shrinks, the estimator's
   variance explodes, and $KDA_{disc}$'s denominator can hit zero outright.
2. **Prior-driven inflation.** Even *away* from saturation, the with-fact term
   $P(R^{q+f}{=}1)$ is contaminated. A solver can be uncertain without the passage
   (giving the question a healthy weight) and still answer with the passage for reasons
   unrelated to the passage — option-surface shortcuts, topical elimination, or a partially
   recalled prior that the passage's *topic* alone is enough to activate.

**A minimal, sufficient fix.** Intervene on the context and observe whether the answer
moves. For each question we build a **counterfactual passage** $f'$ in which the gold answer
string is rewritten as one of the question's own distractors, leaving everything else
untouched. Let $c$ be the counterfactual target option. Then define, per solver $k$:

$$\text{context-dependent}(k, q) \;\equiv\; \left[\arg\max P\left(R_k^{q+f'}\right) = c\right]$$

and three adjusted estimators:

$$KDA_{adj}^{hard}(q) = \frac{\sum_{k} P\left(R_k^{q}=0\right) \, P\left(R_k^{q+f}=1\right) \cdot \mathbb{1}\left[\arg\max P\left(R_k^{q+f'}\right) = c\right]}{\sum_{k} P\left(R_k^{q}=0\right)}$$

$$KDA_{adj}^{soft}(q) = \frac{\sum_{k} P\left(R_k^{q}=0\right) \, P\left(R_k^{q+f}=1\right) \cdot P\left(R_k^{q+f'} = c\right)}{\sum_{k} P\left(R_k^{q}=0\right)}$$

$$KDA_{adj}^{excl} = \underset{q \,\in\, \mathcal{Q}_{\text{ctx}}}{\mathbb{E}}\left[KDA_{cont}(q)\right], \qquad \mathcal{Q}_{\text{ctx}} = \\{q : q \text{ is not ensemble prior-dependent}\\}$$

The **hard** variant keeps only the with-fact probability mass of solvers that provably
follow the context; the **soft** variant weights it by how much mass moves to the
counterfactual target; the **exclusion** variant is a question-level filter that drops
prior-dependent items entirely. Each answers a slightly different question, and Section 6
reports all three.

> **The exclusion variant has two readings, and both are now reported.** The expectation
> above is over $\mathcal{Q}_{\text{ctx}}$ — a mean over the *retained* set, denominator
> $|\mathcal{Q}_{\text{ctx}}|$. The implementation originally reported a different
> quantity: a mean over the *eligible* set with dropped items zero-filled, denominator
> $n_{\text{eligible}}$. They are related exactly,
>
> $$KDA^{\text{excl}}_{\text{zero-filled}} = KDA^{\text{excl}}_{\text{retained}} \cdot \frac{n_{\text{kept}}}{n_{\text{eligible}}},$$
>
> so the published figure was the retained mean scaled by the keep rate. The zero-filled
> form is a **mass-retention statistic** — what share of the ensemble's total KDA mass
> survives the filter — and it is the only one of the two that shares a denominator with
> the hard and soft variants, so only it belongs in a retention comparison against them.
> The retained mean is the filtered *score*. §6.3 reports both; the divergence between
> them is itself a finding.

We also report a **context-verified accuracy**, which replaces raw with-fact accuracy by
counting only correctness that survives the intervention:

$$\text{Acc}^{\text{verified}}_{wf} = \frac{\left|\text{wrong} \to \text{correct}\right| + \left|\text{both-correct} \wedge \text{context-dependent}\right|}{n_{\text{eligible}}}$$

### 2.5. The three-setting experimental design

| Setting | Prompt | Measures |
|---|---|---|
| **A** | question + options | $P(R^q = 1)$ — the parametric prior |
| **B** | question + options + gold passage | $P(R^{q+f} = 1)$ — prior *plus* context |
| **C** | question + options + **counterfactual** passage | which source the solver actually followed |

Every sample that is correct in both A and B — the `both_correct` bucket, invisible to
baseline KDA — is classified by its Setting C behaviour:

| Class | Setting C prediction | Interpretation |
|---|---|---|
| `context_dependent` | flips to the counterfactual target | reads the context; its B-correctness is **earned** |
| `prior_dependent` | stays on the gold answer | overrides context with a prior or shortcut; B-correctness is **not** evidence of knowledge dependence |
| `unstable_other` | moves to a third option | the perturbation broke the solver without steering it |

---

## 3. Related Works & Key Inspirations

### 3.1. KDA — Moon et al., EMNLP 2022

*Evaluating the Knowledge Dependency of Questions.* [`docs/papers/KDA_Paper_Documentation.md`](docs/papers/KDA_Paper_Documentation.md)

The direct baseline and the metric we reproduce in Experiment 1. Introduces reference-free
knowledge-dependent answerability, validated against 116 students and 7 secondary-school
science teachers, and shows $KDA_{disc}$/$KDA_{cont}$ correlating with human KDA at
0.80/0.74 Pearson versus 0.21–0.27 for BLEU/ROUGE/METEOR. Figure 4 of that paper reports
that 82% of questions scoring $KDA_{cont} > 0.8$ were accepted by expert teachers for
classroom use — the practical filtering claim our RQ3 revisits.

**What we take:** the metric, the $KDA_{small}$ four-model suite, and the entire
two-setting scoring apparatus.

**What we push on:** Limitations §6.2 (*too easy questions*) and §6.3 (*difficulty of
measuring PLM's ignorance*). The paper names the problem and defers it; we operationalize a
measurement for it. Their Table 11 already shows the symptom — SciQ and TabMCQ reach
0.96–0.99 average with-fact correctness, near ceiling — and their Table 14 catalogues
"High KDA, Low Likert" cases that are near-paraphrases of the fact.

### 3.2. ClashEval — Wu, Wu & Zou, NeurIPS 2024

*Quantifying the tug-of-war between an LLM's internal prior and external evidence.*
[`docs/papers/ClashEval_Paper_Documentation.md`](docs/papers/ClashEval_Paper_Documentation.md)

The methodological inspiration for our Setting C. ClashEval builds 1,294 questions across
six domains, elicits each model's **prior response** with no context, then **systematically
perturbs** the reference document and re-asks — measuring which source the model followed.
Headline result: GPT-4o adopts *incorrect* retrieved content over its own correct prior
**60.8%** of the time. Context preference falls as the perturbation grows more blatant, and
falls as the model's prior token-probability rises.

**What we take:** the perturb-the-context-and-see-who-wins design, and the vocabulary of
**prior bias** vs. **context bias**:

$$\text{Prior Bias} = P\big(r(q|c) \text{ wrong} \mid c \text{ right}, r(q) \text{ wrong}\big), \quad \text{Context Bias} = P\big(r(q|c) \text{ wrong} \mid c \text{ wrong}, r(q) \text{ right}\big)$$

**What we change:** ClashEval studies this as a *RAG safety* problem — bad retrieval
poisoning good answers. We invert the framing and use the same intervention as a
*measurement instrument for question quality*. Where ClashEval treats a model following a
perturbed context as a failure, we treat it as **evidence that the question is genuinely
knowledge-dependent**. Their perturbations are also graded (0.1×–10× on numerics); ours are
deliberately **minimal and lexical** — a single answer-string substitution — so that
Settings B and C differ in exactly one respect.

### 3.3. QG-SMS — Nguyen et al., ACL 2025

*Enhancing Test Item Analysis via Student Modeling and Simulation.*
[`docs/papers/QG-SMS_Paper_Documentation.md`](docs/papers/QG-SMS_Paper_Documentation.md)

The blueprint for RQ3. QG-SMS imports **test item analysis** — topic coverage (TC), item
difficulty (DF), item discrimination (DC), distractor efficiency (DE) — into QG evaluation,
and shows existing evaluators score ~95.6% on the pre-examination dimension (TC) but
collapse on the post-examination ones: **DF 49.1%, DC 44.5%, DE 53.3%**. Its three-step
pipeline (generate diverse student profiles → predict their per-item performance → judge)
closes that gap.

**Why it matters here.** QG-SMS's motivating case study is precisely our RQ1 in a different
guise: an LLM judge rates an "apply-level" question as more discriminating than a
"recall-level" one, and actual student data says the opposite — because *the apply-level
content was common knowledge*. Common knowledge defeating a content-only evaluator is the
same failure our counterfactual test catches. QG-SMS also reports (Table 4) that KDA remains
the strongest individual-scoring baseline for distractor efficiency (Spearman 0.43),
which is why a corrected KDA is worth having.

**What we take:** the item-analysis dimensions as validation targets, the simulated-student
methodology, and the pairwise-with-order-swap evaluation protocol (Average Accuracy vs.
Consistent Accuracy) for Experiment 3.

### 3.4. Positioning

| | Measures knowledge dependency | Tests prior vs. context | Grounded in student performance |
|---|:---:|:---:|:---:|
| BLEU / ROUGE / METEOR | ✗ | ✗ | ✗ |
| KDA (EMNLP 2022) | ✓ | ✗ | via human study |
| ClashEval (NeurIPS 2024) | ✗ | ✓ | ✗ |
| QG-SMS (ACL 2025) | partial | ✗ | ✓ (simulated) |
| **This work** | ✓ | ✓ | 🔜 Experiment 3 |

---

## 4. Repository Structure & Directory Map

```
experiment/
├── README.md                                  this document
├── REPRODUCE.md                               full step-by-step reproduction guide
│
├── code/
│   ├── utils/                                 shared, stage-agnostic helpers
│   │   ├── __init__.py
│   │   └── paths.py                           PROJECT_ROOT anchoring, resolve(), ensure_parent()
│   │
│   ├── pre_data/                              STAGE 0 — dataset preparation
│   │   ├── __init__.py
│   │   ├── prepare_sciq.py                    allenai/sciq        -> KDA input format
│   │   ├── prepare_openbookqa.py              allenai/openbookqa  -> KDA input format (joins fact1)
│   │   └── download_models.py                 fetch LLM weights into models/ + SHA-256 verify
│   │
│   ├── ex1_reproduce_KDA/                     EXPERIMENT 1 — KDA baseline reproduction
│   │   ├── __init__.py
│   │   ├── kda_tiny.py                        SimulatedStudent + KDATiny; the KDA_cont ensemble
│   │   ├── run_experiment.py                  Setting A/B runner over any prepared dataset
│   │   ├── categorize_kda_results.py          four-bucket contingency analysis + prior_share
│   │   ├── kda_qwen_eval.py                   KDA with a local 4-bit Qwen3-4B as the student
│   │   └── run_qwen7b_eval.py                 standalone Qwen2.5-7B runner for Colab / Kaggle
│   │
│   └── ex2_counterfactual/                    EXPERIMENT 2 — counterfactual perturbation
│       ├── __init__.py
│       ├── counterfactual_passage.py          minimal lexical answer -> distractor rewriting
│       └── run_counterfactual_experiment.py   Setting A/B/C sweep + adjusted KDA
│
├── models/                                    downloaded LLM weights + tokenizers
│   ├── Qwen3-4B-Instruct-2507/                local 4-bit evaluation target (7.5GB)
│   └── Qwen2.5-7B-Instruct/                   cloud evaluation target (~15GB)
│
├── datasets/                                  prepared, KDA-formatted corpora
│   ├── sciq/                                  sciq_{train,val,test}_full.json,
│   │                                          sciq_all_combined.json, sciq_50.json
│   └── openbookqa/                            obqa_{train,val,test}_full.json,
│                                              obqa_all_combined.json, obqa_50.json
│
├── docs/                                      paper documentation & analysis
│   ├── KDA_Paper_Documentation.md             Moon et al., EMNLP 2022
│   ├── ClashEval_Paper_Documentation.md       Wu et al., NeurIPS 2024
│   ├── QG-SMS_Paper_Documentation.md          Nguyen et al., ACL 2025
│   ├── kda_reproduction_summary.md            SciQ vs OpenBookQA baseline comparison
│   ├── kda_qwen3_4b_evaluation_report.md      Qwen3-4B (4-bit) KDA run — RQ1 at LLM scale
│   └── RUN_QWEN2.5.md          Colab / Kaggle guide for Qwen2.5-7B
│
├── paper_references/                          source PDFs for the three papers above
│
├── results/                                   outputs and logs, grouped by workflow stage
│   ├── ex1_reproduce_KDA_pipeline/            EXPERIMENT 1 — KDA_small / KDA_tiny baselines
│   │   ├── sciq/                              SciQ test outputs
│   │   │   ├── results_kda_small_sciq_test_full.json      884 q, 4-model KDA_small (355M)
│   │   │   ├── results_kda_tiny2_sciq_test_full.json      884 q, 2-model KDA_tiny
│   │   │   └── results_kda_tiny2_sciq_test_sample50.json  the original 50-question run
│   │   ├── openbookqa/                        OpenBookQA test (500 q) outputs
│   │   ├── prep_sciq.log                      SciQ preparation log
│   │   └── prep_openbookqa.log                OpenBookQA preparation log
│   ├── ex1_reproduce_KDA_w_modernLLM/         EXPERIMENT 1b — modern-LLM solvers
│   │   ├── kda_qwen3_4b_results.json          Qwen3-4B (4-bit) on OBQA + SciQ
│   │   ├── kda_qwen3_4b_eval.log              its execution log
│   │   └── model_download_report.json         SHA-256 verification of every weight shard
│   ├── ex1_category_questions/                question categorisations
│   │   ├── basic_category/                    four-bucket breakdown, per model
│   │   └── complicated_category/              reserved for richer categorisations
│   └── ex2_counterfactual/                    EXPERIMENT 2 — counterfactual perturbation
│       ├── results_counterfactual_sciq_test_full.json  Settings A/B/C + adjusted KDA
│       └── counterfactual_passages_preview_sciq.json   sampled rewrites, for audit
│
├── question-score/                            reference KDA implementation (untouched)
└── .venv/
```

### 4.1. Path and import conventions

Every script is **location-independent and working-directory-independent**:

* [`code/utils/paths.py`](code/utils/paths.py) derives `PROJECT_ROOT` from its own file
  location (`code/utils/paths.py` → `code/utils/` → `code/` → `experiment/`). No absolute
  system path is ever hardcoded.
* All `--data`, `--out` and `--log-file` arguments are resolved against `PROJECT_ROOT` via
  `resolve()`, so `--data datasets/sciq/sciq_test_full.json` means the same thing whether
  you run from the repo root, from inside `code/ex2_counterfactual/`, or from `/tmp`.
  Absolute paths passed on the command line are honoured as-is.
* Cross-stage imports go through a short bootstrap that puts `code/` on `sys.path`, making
  `utils.paths` and `ex1_reproduce_KDA.kda_tiny` importable from any stage. The **project
  root is deliberately *not* added** — it contains a `datasets/` directory that would
  otherwise shadow the HuggingFace `datasets` package.
* `code/ex2_counterfactual/` reuses `ex1_reproduce_KDA.kda_tiny.make_student`, so Settings A
  and B in Experiment 2 are produced by exactly the same code path as the Experiment 1
  baseline and are directly comparable.

---

## 5. Quickstart & Execution Guide

> Verified on Ubuntu (Linux 7.0.0-30-generic), Python 3.12.3, NVIDIA RTX 3050 4GB.
> **A GPU is optional** — the scripts fall back to CPU automatically.
> For the exhaustive version of this guide, including every CLI flag and the full set of
> per-question findings, see [`REPRODUCE.md`](REPRODUCE.md).

### Step 0 — Requirements

* Python ≥ 3.9 (verified on 3.12.3)
* Internet access on first run (HuggingFace Hub checkpoints + datasets)
* ~8GB free disk space (the torch CUDA wheels dominate this)

### Step 1 — Environment setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
```

```bash
git clone https://github.com/riiid/question-score.git
```

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
```

```bash
uv pip install torch transformers datasets accelerate protobuf
```

`protobuf` is required only by `google/t5-small-ssm-nq`, whose SentencePiece vocabulary is
converted at load time.

```bash
uv pip install -e ./question-score
```

**Required compatibility patch.** `question-score` targets the 2022-era `transformers` and
imports `AdamW`, a symbol removed in `transformers` v5. It is never used, so deleting it
from the import line is sufficient:

```bash
sed -i 's/^from transformers import AutoModelForMultipleChoice, AutoTokenizer, AdamW$/from transformers import AutoModelForMultipleChoice, AutoTokenizer  # AdamW: unused, removed in transformers v5/' question-score/src/question_score/kda.py
```

Verify:

```bash
python -c "import question_score, torch, transformers; print('question_score OK |', 'torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

> `code/ex1_reproduce_KDA/kda_tiny.py` does **not** import `question_score` — it
> reimplements the scoring logic — so this patch is only needed to run the reference
> implementation for comparison.

### Step 2 — Dataset preprocessing

Export every SciQ split in the KDA input format (`passage`, `question`, `options`,
`answer_idx`), writing to `datasets/sciq/`:

```bash
python code/pre_data/prepare_sciq.py
```

Same for OpenBookQA, joining `fact1` from the `additional` config as the target fact:

```bash
python code/pre_data/prepare_openbookqa.py
```

Draw a small fixed-size subset for smoke tests:

```bash
python code/pre_data/prepare_sciq.py --mode sample --n 50 --split test --out datasets/sciq/sciq_50.json
```

Both preparers are **deterministic and non-destructive**, and write per-split counts and
drop reasons to `results/ex1_reproduce_KDA_pipeline/prep_sciq.log` and `results/ex1_reproduce_KDA_pipeline/prep_openbookqa.log`.

### Step 3 — Experiment 1: reproduce the KDA baseline

Full SciQ test split with the official four-model `KDA_small` suite (Settings A and B):

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_test_full.json --models KDA_SMALL --out results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json --log-file results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_small_sciq_test_full.log --progress-every 250
```

OpenBookQA, for the cross-dataset comparison:

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/openbookqa/obqa_test_full.json --models KDA_SMALL --dataset-name allenai/openbookqa --split test --out results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json --log-file results/ex1_reproduce_KDA_pipeline/openbookqa/experiment_kda_small_obqa_test_full.log --progress-every 250
```

Break the run into the four contingency buckets (`both_correct`, `wrong_to_correct`,
`both_wrong`, `correct_to_wrong`) and report each model's `prior_share`:

```bash
python code/ex1_reproduce_KDA/categorize_kda_results.py --results results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json --out-dir results/ex1_category_questions/basic_category
```

> **Smoke test first.** A two-model, 50-question run finishes in seconds on CPU:
> `python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_50.json --models Riiid/kda-distilbert-base-uncased-race google/t5-small-ssm-nq --device cpu --out results/ex1_reproduce_KDA_pipeline/smoke.json --log-file results/ex1_reproduce_KDA_pipeline/smoke.log`

### Step 4 — Experiment 2: counterfactual evaluation

Sweep Settings A, B **and C** over the full SciQ test split with all four `KDA_small`
models, and compute the adjusted metrics:

```bash
python code/ex2_counterfactual/run_counterfactual_experiment.py --data datasets/sciq/sciq_test_full.json --out results/ex2_counterfactual/results_counterfactual_sciq_test_full.json --log-file results/ex2_counterfactual/counterfactual_sciq_test_full.log
```

Quick smoke test on 20 questions with a single model:

```bash
python code/ex2_counterfactual/run_counterfactual_experiment.py --limit 20 --models Riiid/kda-mpnet-base-race --device cpu --out results/ex2_counterfactual/cf_smoke.json --log-file results/ex2_counterfactual/cf_smoke.log
```

Restrict the analysis to the cleanest lexical substitutions only (drops `partial` and
`morphological` tiers):

```bash
python code/ex2_counterfactual/run_counterfactual_experiment.py --min-substitution-tier exact
```

Experiment 2 sweeps models **one at a time** by construction (there is no `--strategy`
flag here) because the `KDA_small` suite does not fit in 4GB of VRAM together; each model runs all
three settings over the whole dataset, then its weights are released.

### Step 5 — LLM solvers (optional)

Download the target LLM weights into `models/` and verify every shard by SHA-256:

```bash
uv run --active python code/pre_data/download_models.py
```

Run the KDA evaluation with `Qwen3-4B-Instruct-2507` under 4-bit NF4 quantisation — this
fits a 4GB GPU and takes about 8 minutes for both test splits:

```bash
uv run --active python code/ex1_reproduce_KDA/kda_qwen_eval.py --datasets obqa sciq --out results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json --log-file results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_eval.log
```

Results are written to `results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json`; the analysis is in
[`docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md`](docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md).

`Qwen2.5-7B-Instruct` does not fit a 4GB GPU. To run it on Colab or Kaggle, follow
[`RUN_QWEN2.5.md`](RUN_QWEN2.5.md), which wraps
[`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](code/ex1_reproduce_KDA/run_qwen7b_eval.py).

### 5.1. Useful flags

| Flag | Applies to | Purpose |
|---|---|---|
| `--models KDA_SMALL` | ex1, ex2 | Expand to the paper's official four-model suite |
| `--device cpu` \| `cuda` | ex1, ex2 | Override auto-detection |
| `--strategy resident` \| `sequential` | ex1 | Keep all models in VRAM vs. sweep one at a time |
| `--limit N` | ex2 | Score only the first N samples |
| `--primary-model` | ex2 | Model whose buckets drive the reported target set |
| `--dry-run` | ex2 | Build the counterfactual passages and report tier stats without loading any model |
| `--min-substitution-tier` | ex2 | `exact` \| `glued` \| `morphological` \| `partial` |
| `--progress-every N` | ex1, ex2 | Progress line cadence (0 = automatic) |
| `--verbose` | all | Stream DEBUG records to the console |
| `--append-log` | all | Append instead of overwriting the log |
| `--double-quant` | `kda_qwen_eval` | Nested 4-bit quantisation; frees ~0.1GB VRAM |
| `--datasets obqa sciq` | `kda_qwen_eval`, `run_qwen7b_eval` | Which test splits to score |
| `--verify-only` | `download_models` | Re-check checksums without re-downloading |
| `--precision` | `run_qwen7b_eval` | `auto` \| `bf16` \| `fp16` \| `8bit` \| `4bit` |

Every log records the Python/torch/platform versions, the resolved absolute input and
output paths, ensemble membership, per-model VRAM before and after load, and a DEBUG line
per question per model carrying both full 4-way probability vectors — enough to recompute
every reported number without re-running any model.

---

## 6. Findings Pipeline & Current Results

### 6.1. Pipeline

```
  ┌────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐    ┌───────────────┐
  │  STAGE 0       │    │  EXPERIMENT 1       │    │  EXPERIMENT 2        │    │ EXPERIMENT 3  │
  │  pre_data/     │──▶ │  ex1_reproduce_KDA/ │──▶ │  ex2_counterfactual/ │──▶ │  (planned)    │
  ├────────────────┤    ├─────────────────────┤    ├──────────────────────┤    ├───────────────┤
  │ SciQ, OBQA     │    │ Settings A + B      │    │ Setting C            │    │ Filter/rerank │
  │ → KDA format   │    │ KDA_cont baseline   │    │ prior vs. context    │    │ quiz banks    │
  │ passage,       │    │ 4-bucket analysis   │    │ adjusted KDA         │    │ human/teacher │
  │ question,      │    │ ↳ RQ1 symptom       │    │ ↳ RQ1 measured       │    │ validation    │
  │ options,       │    │                     │    │ ↳ RQ2 method         │    │ ↳ RQ3         │
  │ answer_idx     │    │                     │    │                      │    │               │
  └────────────────┘    └─────────────────────┘    └──────────────────────┘    └───────────────┘
```

### 6.2. Experiment 1 — the baseline reproduces, and shows the symptom

SciQ `test` (884 questions) and OpenBookQA `test` (500 questions), `KDA_small` suite
(|M| = 4, 355.1M parameters), CUDA, identical code path for both.

| | SciQ (884 q) | OpenBookQA (500 q) |
|---|---:|---:|
| **Mean $KDA_{cont}$** | **0.4715** | **0.2712** |
| Pooled-vote Acc without fact | 45.48% | 34.80% |
| Pooled-vote Acc with fact | 88.57% | 40.60% |
| Pooled-vote Δ | +43.10% | +5.80% |
| Mean $P(R^q{=}1) \to P(R^{q+f}{=}1)$ | 0.331 → 0.530 | 0.285 → 0.315 |

The −42.5% gap between the two datasets is a *mixture effect*, not a defect: the correct
answer appears verbatim in the SciQ `support` passage for **820/884 questions (92.8%)**,
making Setting B largely an extraction task, while OpenBookQA's one-clause `fact1` contains
the answer verbatim in only **47/500 (9.4%)**, leaving a genuine deduction task. Partitioning
each dataset by verbatim containment confirms the mechanism directly (SciQ: 0.4806 vs.
0.3548; OBQA: 0.3725 vs. 0.2607).

**Reproduction integrity is verified.** The OpenBookQA baseline was executed twice end to
end: all 4,000 probability vectors, all argmax predictions and all 500 `kda_score` values
are **identical at full float precision** (`mean_kda_tiny = 0.27116560020399944` both
times). Only wall-clock timings differ. Full detail in
[`docs/ex1_reproduce_KDA/kda_reproduction_summary.md`](docs/ex1_reproduce_KDA/kda_reproduction_summary.md).

**The symptom.** On SciQ, `both_correct` — questions the solver answers correctly *with and
without* the passage — is the single largest bucket for the strongest model (mpnet: 423
questions, 47.9%). Baseline KDA cannot see inside it.

### 6.3. Experiment 2 — measuring the finding (RQ1) with the new method (RQ2)

SciQ `test`, 884 questions, of which **860 (97.3%)** admit a valid counterfactual rewrite
(substitution tiers: 813 `exact`, 36 `partial`, 11 `morphological`, 24 ineligible).

**Inside the `both_correct` bucket**, Setting C splits the samples as follows:

| Model | `both_correct` (n) | context-dependent | **prior-dependent** | unstable | Acc$_{wf}$ inflation |
|---|---:|---:|---:|---:|---:|
| `google/t5-small-ssm-nq` | 302 | 30.5% | **63.9%** | 5.6% | **+22.4 pp** |
| `Riiid/kda-scibert-uncased-race` | 255 | 65.5% | **31.0%** | 3.5% | +9.2 pp |
| `Riiid/kda-albert-xlarge-v2-race` | 157 | 62.4% | **22.3%** | 15.3% | +4.1 pp |
| `Riiid/kda-mpnet-base-race` | 415 | 90.1% | **8.2%** | 1.7% | +4.0 pp |

**This is RQ1, measured.** Between 8% and 64% of the with-fact correct answers that baseline
KDA credits to the target material are produced *without* it. The closed-book
NQ-finetuned T5 — the solver in the suite with the most parametric world knowledge — is by
far the worst offender: it keeps answering the gold option in **63.9%** of cases where the
passage has been rewritten to assert a distractor, inflating its apparent with-fact accuracy
by **22.4 percentage points**. The pattern is monotone in the direction the theory predicts:
*the more a solver knows a priori, the less its with-fact correctness means.* Extrapolating
that trend to a modern LLM is precisely the failure mode RQ1 names.

**And this is RQ2, applied.** Correcting the metric:

| Estimator | Mean | Retention vs. baseline |
|---|---:|---:|
| $KDA_{cont}$ (860 CF-eligible questions) | 0.4767 | — |
| $KDA_{adj}^{hard}$ | **0.3357** | 70.4% |
| $KDA_{adj}^{soft}$ | **0.2641** | 55.4% |
| $KDA_{adj}^{excl}$, zero-filled / mass-retention (94/860 dropped) | **0.4269** | 89.6% |
| $KDA_{adj}^{excl}$, retained mean — §2.4's $\mathbb{E}_{q \in \mathcal{Q}_{ctx}}$ form | **0.4793** | **100.5%** |

The intervention strips **29.6%** of the baseline score under the hard adjustment and
**44.6%** under the soft one — i.e. only about seven-tenths (resp. just over half) of the
knowledge dependency SciQ appears to have survives a check that the answer actually came
from the passage.

**The two exclusion rows are the more interesting result.** Read as §2.4 defines it — a
mean over the questions that survive the filter — the exclusion variant is **inert**: it
returns a score 0.5% *higher* than the unfiltered baseline. Dropping the 94 questions whose
Setting-B correctness Setting C fails to justify does not lower the mean, because those
questions are not low-KDA questions. Their mean $KDA_{cont}$ is 0.4554 against the retained
set's 0.4793, and $KDA_{cont}$'s ability to tell the two groups apart is near chance (AUC
**0.553**; on OpenBookQA `partial` it is 0.346, i.e. *inverted*). **$KDA_{cont}$ carries almost no information about whether a
question is answered from prior or from context** — which is the RQ1 thesis restated from
inside the estimator meant to repair it.

That conclusion is not an artifact of how $\mathcal{Q}_{ctx}$ is drawn. Across nine
definitions — ensemble vote, all-buckets, each of the four solvers individually, and
any/majority/all-member votes — dropping between 0 and 337 of the 860 questions, the
retained mean stays within **98.4%–103.7%** of baseline (95.2%–110.7% across all three
runs). Full sweep: [`results/ex2_counterfactual/adjusted_kda_corrected.json`](results/ex2_counterfactual/adjusted_kda_corrected.json).

The zero-filled row measures something real but different — mass retention — and on that
common denominator the comparison against the hard figure holds: only 9.9% of questions are
prior-dependent for the *ensemble as a whole*, while far more are prior-dependent for *some
member*, so ensemble-vote question-level filtering is less sensitive than per-solver
adjustment. That ordering is a property of the ensemble-vote definition rather than of the
two families: under an any-member $\mathcal{Q}_{ctx}$, question-level filtering drops 337
questions and falls to 60.0% mass retention, below the hard variant's 70.4%.

> **Scoring convention.** All four estimators treat `unstable_other` — a Setting-C
> prediction that is neither the gold answer nor the counterfactual target — as earning no
> credit. Before 2026-09-09 the exclusion variant alone retained those items at full
> $KDA_{cont}$, disagreeing with the other three. The choice is not cosmetic: it moves hard
> retention by up to 29 pp on OpenBookQA. The decision, the alternatives, and the numbers
> are in [`docs/ex2_counterfactual/unstable_other_convention.md`](docs/ex2_counterfactual/unstable_other_convention.md).

Per-model **context-verified accuracy** replaces the headline with-fact numbers:

| Model | Raw Acc$_{wf}$ | Context-verified Acc$_{wf}$ |
|---|---:|---:|
| `google/t5-small-ssm-nq` | 63.8% | **39.4%** |
| `Riiid/kda-scibert-uncased-race` | 69.0% | **58.7%** |
| `Riiid/kda-albert-xlarge-v2-race` | 63.5% | **56.6%** |
| `Riiid/kda-mpnet-base-race` | 90.6% | **85.8%** |

### 6.4. Experiment 1b — RQ1 at modern-LLM scale

`Qwen3-4B-Instruct-2507` under 4-bit NF4 quantisation on a 4GB RTX 3050, scoring the same
two test splits. Full write-up: [`docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md`](docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md).

| Dataset | $n$ | $KDA_{disc}$ | $KDA_{cont}$ | $Acc_{wof}$ | $Acc_{wf}$ | Memorisation |
|---|---:|---:|---:|---:|---:|---:|
| OpenBookQA | 500 | 0.5862 | 0.5956 | 82.60% | 92.00% | 82.60% |
| SciQ | 884 | 0.9512 | 0.9407 | 95.36% | 99.77% | 95.36% |

Against the `KDA_small` encoder ensemble on the identical files, without-fact accuracy
roughly doubles — SciQ **45.48% → 95.36%**, OpenBookQA **34.80% → 82.60%** (pooled vote).
Three findings follow, and all three sharpen RQ1:

1. **The metric's support collapses.** $KDA_{disc}$'s denominator is the count of questions
   answered *wrong* without the fact. On SciQ that is **41 of 884 (4.6%)**; on OpenBookQA
   **87 of 500 (17.4%)**. The reported 0.9512 is a confident-looking number computed from
   4.6% of the data, and the metric emits no signal that this happened. $KDA_{cont}$'s
   continuous denominator does not help: 40.46 out of a possible 884.
2. **$KDA_{cont}$ becomes redundant.** Restricted to four option letters the model's
   softmax saturates — 777/884 SciQ samples have $P(R^q{=}1)$ equal to `1.0` in float32,
   and only 2 samples fall in $[0.25, 0.90)$. The continuous relaxation exists to recover
   graded information; there is none left, and it collapses onto $KDA_{disc}$ (0.9407 vs
   0.9512).
3. **Both of the metric's blind spots sit on the $r^q{=}1$ side.** `both_correct` (843 SciQ,
   409 OBQA — pure parametric prior) and `correct_to_wrong` (4 OBQA — cases where the fact
   *hurt*, ClashEval's context bias appearing inside the KDA construction) are both silently
   dropped. As solver capability rises that side grows, so the fraction of the dataset the
   metric can see shrinks toward zero.

This is the RQ1 extrapolation in the earlier sections, now measured rather than predicted.
It also motivates Experiment 2's intervention more strongly: when 95% of a dataset is
invisible to the baseline metric, a counterfactual test of *why* an answer was correct
stops being a refinement and becomes the only available signal.

> **Aggregation caveat.** These KDA values aggregate over *samples* at the dataset level;
> §6.2's aggregate over *models* per question. The two are different quantities and are
> never compared numerically — only the accuracies above are directly comparable.

### 6.5. Experiment 3 — planned

Apply the Section 2.4 estimators as a filter and reranker over generated quiz banks and
validate downstream:

* **Arms:** *No Filtering* vs. *KDA Filtering* ($KDA_{cont}$ threshold, following the
  original paper's 0.8 operating point) vs. *Our Disentangled Filtering*
  ($KDA_{adj}^{hard}$, or $KDA_{cont}$ restricted to $\mathcal{Q}_{\text{ctx}}$).
* **Validation:** human/teacher evaluation that students genuinely require the course
  material to answer surviving questions, on the QG-SMS item-analysis dimensions (topic
  coverage, difficulty, discrimination, distractor efficiency), scored with both Average
  Accuracy and order-swap Consistent Accuracy.
* **Scaling to modern LLMs — answered.** Setting C has now been run with `Qwen3-4B` on SciQ
  ([`docs/ex2_counterfactual/e1_counterfactual_llm_scale.md`](docs/ex2_counterfactual/e1_counterfactual_llm_scale.md)). The
  intervention **does** still separate prior from context on a saturated solver: on a target
  set of **819** items (2.15× the encoder run's, because saturation grows the bucket baseline
  KDA cannot see), **62.5%** are `prior_dependent` [95% CI 59.2–65.8] against the encoder
  ensemble's 22.3%, and context-verified Acc$_{wf}$ falls from a raw **0.9977** to **0.3988**
  — a prior inflation of **+59.9 pp** versus +4.0 pp for the encoders.

  Two findings constrain how that can be used. The Setting-C label agrees across the two
  solvers at only **κ = 0.046**, so it is a property of the solver rather than the question —
  a direct problem for the *Our Disentangled Filtering* arm below, which assumes such labels
  can filter a question bank; that remains open. The second — that `prior_dependent` might
  just be the model *correctly rejecting a false* context — was tested and largely ruled out
  ([`docs/ex2_counterfactual/e2_prior_vs_rejection.md`](docs/ex2_counterfactual/e2_prior_vs_rejection.md)): only 26.4% of those
  items follow the passage even when ordered to, and the model's own plausibility penalty
  does not predict which refuse (AUC 0.496). Genuine prior-dependence is bracketed at
  **46.0%–62.5%**, against the encoder ensemble's 22.3%.
  [`RUN_QWEN2.5.md`](RUN_QWEN2.5.md) covers scaling
  to `Qwen2.5-7B-Instruct` on Colab / Kaggle, which would give the first non-degenerate
  |M| = 2 LLM ensemble and a second reading of that κ.

### 6.6. Known caveats

* **Substitution is lexical, not semantic.** The counterfactual rewrite is a whole-word
  answer-string substitution. 24/884 SciQ questions (2.7%) have no lexical match and are
  excluded from every Setting C metric; 16 samples retain a "glued" residual mention of the
  gold answer. Restrict to `--min-substitution-tier exact` for the cleanest subset.
* **`unstable_other` is scored as a failure, and that is a judgement call.** Between 1.7%
  and 15.3% of `both_correct` samples move to a third option under Setting C — the
  perturbation disturbed the solver without steering it. All four estimators now credit
  these to neither class (the **strict** convention). At the (model, question) pair level
  the exposure is larger than those figures suggest — 19.7% on SciQ, 34.5% on OBQA `exact` —
  and it is reported with every run as `unstable_other_exposure`. The convention cannot
  separate "the solver broke" from "the perturbation was incoherent", so on OBQA these
  numbers are depressed by perturbation quality as well as by solver behaviour. Rationale
  and alternatives: [`docs/ex2_counterfactual/unstable_other_convention.md`](docs/ex2_counterfactual/unstable_other_convention.md).
* **Dataset dependence.** SciQ's extractive passages make Setting C a comparatively easy
  intervention. OpenBookQA's one-clause deductive facts do not support the same lexical
  rewrite at scale; extending the intervention there is open work.
* **Cross-dataset $KDA$ values are not on a common scale.** Any comparison of absolute KDA
  across datasets must control for target-fact style.
* **Naming caveat.** The `KDA_small` suite is published under the HuggingFace org `Riiid`
  (three i's). Two names cited in the original paper do not resolve on the Hub:
  `Riiid/kda-scibert-scivocab-uncased-race` (the real name is
  `Riiid/kda-scibert-uncased-race`) and `Riiid/kda-albert-base-v2-race` (only the xlarge and
  xxlarge ALBERT variants were released).

---

## References

* Moon, H., Yang, Y., Shin, J., Yu, H., Lee, S., Jeong, M., Park, J., Kim, M., & Choi, S.
  (2022). *Evaluating the Knowledge Dependency of Questions.* EMNLP 2022, 10512–10526.
  [`paper_references/2022.emnlp-main.718.pdf`](paper_references/2022.emnlp-main.718.pdf) ·
  [riiid/question-score](https://github.com/riiid/question-score)
* Wu, K., Wu, E., & Zou, J. (2024). *ClashEval: Quantifying the tug-of-war between an LLM's
  internal prior and external evidence.* NeurIPS 2024, Datasets and Benchmarks Track.
  [`paper_references/NeurIPS-2024-clasheval-....pdf`](paper_references/NeurIPS-2024-clasheval-quantifying-the-tug-of-war-between-an-llms-internal-prior-and-external-evidence-Paper-Datasets_and_Benchmarks_Track.pdf) ·
  [kevinwu23/StanfordClashEval](https://github.com/kevinwu23/StanfordClashEval)
* Nguyen, B., Du, T., Yu, M., Angrave, L., & Jiang, M. (2025). *QG-SMS: Enhancing Test Item
  Analysis via Student Modeling and Simulation.* ACL 2025 (Long Papers), 26152–26168.
  [`paper_references/2025.acl-long.1268.pdf`](paper_references/2025.acl-long.1268.pdf) ·
  [bnguyen5/qg-sms](https://github.com/bnguyen5/qg-sms)

### Datasets

* `allenai/sciq` — Welbl, Liu & Gardner (2017), *Crowdsourcing Multiple Choice Science Questions.*
* `allenai/openbookqa` — Mihaylov, Clark, Khot & Sabharwal (2018), *Can a Suit of Armor Conduct Electricity?*
