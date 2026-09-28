# KDA Evaluation with `Qwen3-4B-Instruct-2507` (4-bit NF4, RTX 3050 4GB)

Evaluation report for the **KDA** metric computed with a modern instruction-tuned LLM as
the simulated student, on the full `test` splits of **OpenBookQA** (500 questions) and
**SciQ** (884 questions).

This run exists to test the central claim of [the project's RQ1](../../README.md#rq1--finding-a-failure-mode-of-modern-llm-evaluators):
that KDA's load-bearing assumption — *the solver does not already know the answer* — fails
on modern LLMs. **It does, and more severely than the framing anticipated.**

---

## Headline finding

> On SciQ, **only 41 of 884 questions (4.6%)** contribute anything at all to $KDA_{disc}$.
> The other 843 are answered correctly by Qwen3-4B **without ever seeing the target
> passage**, which zeroes their weight $(1 - r^q)$ and removes them from the metric
> entirely. On OpenBookQA the usable support is 87 of 500 (17.4%).

The metric does not fail loudly. It returns a high, confident-looking number
($KDA_{disc} = 0.9512$ on SciQ) computed from **4.6% of the dataset**. A practitioner
reading only the headline score would conclude SciQ's questions are almost perfectly
knowledge-dependent. What the number actually says is that the 41 questions this particular
model happened not to know were mostly answerable once the passage was supplied.

---

## Contents

1. [Model configuration and environment](#1-model-configuration-and-environment)
2. [Memory footprint and latency](#2-memory-footprint-and-latency)
3. [Scoring protocol](#3-scoring-protocol)
4. [Results](#4-results)
5. [Finding 1 — the metric's support collapses](#5-finding-1--the-metrics-support-collapses)
6. [Finding 2 — probability saturation makes KDA_cont redundant](#6-finding-2--probability-saturation-makes-kda_cont-redundant)
7. [Finding 3 — comparison with the KDA_small encoder baseline](#7-finding-3--comparison-with-the-kda_small-encoder-baseline)
8. [Error analysis and qualitative case studies](#8-error-analysis-and-qualitative-case-studies)
9. [Limitations and caveats](#9-limitations-and-caveats)
10. [Reproduction](#10-reproduction)

---

## 1. Model configuration and environment

| | |
|---|---|
| **Model** | `Qwen/Qwen3-4B-Instruct-2507` (local copy at `models/Qwen3-4B-Instruct-2507/`) |
| **Quantisation** | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)` |
| **Double quantisation** | disabled (`bnb_4bit_use_double_quant=False`) — the model fits without it |
| **Packed parameters** | 2,205,810,176 (4-bit packed representation of the 4.02B model) |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU — 4 GB nominal, **3.68 GiB addressable** |
| **Driver / CUDA** | 580.173.02 · torch 2.13.0+cu130 |
| **Libraries** | `transformers` 5.15.1 · `bitsandbytes` 0.50.2 · `accelerate` 1.14.0 |
| **Python / OS** | 3.12.3 · Linux 7.0.0-30-generic (x86_64, glibc 2.39) |
| **Batch size** | 1 (single sequence per forward pass) |
| **Datasets** | `datasets/openbookqa/obqa_test_full.json` (500) · `datasets/sciq/sciq_test_full.json` (884) |
| **Target fact** | OpenBookQA → `fact1` · SciQ → `support` paragraph |

Both datasets were scored in a single process, in one pass, with no sampling: inference
runs under `torch.no_grad()` with greedy logit reads, so the run is deterministic.

## 2. Memory footprint and latency

| Metric | Value |
|---|---:|
| Model load time | 6.29 s |
| **Weights resident in VRAM** | **2.657 GB** |
| Allocated after load | 2.657 GB |
| Reserved after load | 2.703 GB |
| **Peak allocated (whole run)** | **2.988 GB** |
| **Peak reserved (whole run)** | **3.488 GB** |
| Device total (addressable) | 3.680 GB |
| **Headroom at peak** | **0.192 GB (5.2%)** |

4-bit NF4 brings the 4B model to 2.66 GB of weights, leaving ~1 GB for activations. Peak
*reserved* memory reached 3.49 GB of 3.68 GB — 95% utilisation. The run completed without
a single OOM, but the margin is thin: **SciQ's longest support passage produces a 723-token
prompt**, and that is what sets the peak. Enabling `--double-quant` frees a further ~0.1 GB
if you need the headroom.

### Latency

| Dataset | Samples | Scoring time | **Per sample** | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| OpenBookQA | 500 | 146.8 s | **0.293 s** | 0.289 s | 0.272 s | 1.210 s |
| SciQ | 884 | 333.1 s | **0.377 s** | 0.355 s | 0.282 s | 0.794 s |
| **Total** | **1,384** | **479.9 s** | **0.347 s** | — | — | — |

End-to-end runtime including model load: **8.2 minutes**. Each "sample" is *two* forward
passes (without-fact and with-fact), so the per-pass cost is ~0.17 s.

SciQ is 29% slower per sample than OpenBookQA because its target fact is a retrieved
paragraph rather than a one-clause statement:

| Dataset | Prompt tokens (no fact) | Prompt tokens (with fact) |
|---|---|---|
| OpenBookQA | mean 87.8, max 153 | mean 100.5, max 167 |
| SciQ | mean 85.6, max 133 | **mean 188.1, max 723** |

## 3. Scoring protocol

Each sample is scored twice, mirroring the KDA construction:

| Phase | Prompt | Yields |
|---|---|---|
| **1 — without fact** | question + lettered options | $P(R^q = 1)$, $r^q \in \{0,1\}$ |
| **2 — with fact** | gold fact + question + lettered options | $P(R^{q+f} = 1)$, $r^{q+f} \in \{0,1\}$ |

Both phases use a **single forward pass**. The Qwen chat template is applied with
`add_generation_prompt=True`, and the logits at the final position are restricted to the
four option letters. All single-token spellings of each letter are combined with
`logsumexp` before a softmax over the four options:

```
A -> [32, 362]   ('A', ' A')      C -> [34, 356]   ('C', ' C')
B -> [33, 425]   ('B', ' B')      D -> [35, 422]   ('D', ' D')
```

$$P(R = 1) = p_{\text{gold letter}}, \qquad r = \mathbb{1}\left[\arg\max_\ell p_\ell = \text{gold letter}\right]$$

This is standard MMLU-harness multiple-choice scoring. It is *not* per-option sequence
likelihood, which requires four passes and is length-biased; the letter-logit form keeps
every option on equal footing at a quarter of the compute.

### Metrics

Aggregated over samples $i$, as specified for this study:

$$KDA_{disc} = \frac{\sum_{i} (1 - r_i^q)\, r_i^{q+f}}{\sum_{i} (1 - r_i^q)}, \qquad KDA_{cont} = \frac{\sum_{i} P(R_i^q = 0)\, P(R_i^{q+f} = 1)}{\sum_{i} P(R_i^q = 0)}$$

with $P(R^q = 0) = 1 - P(R^q = 1)$.

> **Aggregation caveat.** This is a *dataset-level* aggregation over samples. The encoder
> baseline in [`run_experiment.py`](../../code/ex1_reproduce_KDA/run_experiment.py) computes a
> *per-question* $KDA_{cont}$ aggregated over models $k$, then averages. The two are
> different quantities and their KDA values are **not directly comparable**. Accuracies
> ($Acc_{wof}$, $Acc_{wf}$) *are* directly comparable, and §7 restricts itself to those.

## 4. Results

### 4.1. Headline metrics

| Dataset | $n$ | $KDA_{disc}$ | $KDA_{cont}$ | $Acc_{\text{wof}}$ | $Acc_{\text{wf}}$ | $\Delta$ | Memorisation ($r^q{=}1$) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OpenBookQA** | 500 | 0.5862 | 0.5956 | 82.60% | 92.00% | **+9.40 pp** | **82.60%** |
| **SciQ** | 884 | 0.9512 | 0.9407 | 95.36% | 99.77% | **+4.41 pp** | **95.36%** |

### 4.2. Probability-space view

| Dataset | mean $P(R^q{=}1)$ | mean $P(R^{q+f}{=}1)$ | Gain |
|---|---:|---:|---:|
| OpenBookQA | 0.8276 | 0.9171 | +0.0895 |
| SciQ | 0.9542 | 0.9973 | +0.0431 |

### 4.3. Four-bucket contingency table

| Dataset | `both_correct` | `wrong_to_correct` | `both_wrong` | `correct_to_wrong` |
|---|---:|---:|---:|---:|
| **OpenBookQA** | 409 (81.80%) | 51 (10.20%) | 36 (7.20%) | 4 (0.80%) |
| **SciQ** | 843 (95.36%) | 39 (4.41%) | 2 (0.23%) | 0 (0.00%) |

`prior_share` — the fraction of with-fact-correct answers that were *already* correct
without the fact — is **0.8891** on OpenBookQA and **0.9558** on SciQ. Nine out of ten
(OBQA) to nineteen out of twenty (SciQ) of the model's "the fact made this answerable"
successes were not caused by the fact at all.

## 5. Finding 1 — the metric's support collapses

$KDA_{disc}$'s denominator is $\sum_i (1 - r_i^q)$: the count of questions the solver got
**wrong** without the fact. Every question the model already knows contributes exactly zero
to both numerator and denominator — it is silently dropped.

| Dataset | Denominator | Usable share of dataset | Discarded |
|---|---:|---:|---:|
| **SciQ** | **41 / 884** | **4.64%** | 843 questions |
| **OpenBookQA** | **87 / 500** | **17.40%** | 413 questions |

The continuous form is no better. $KDA_{cont}$'s denominator is $\sum_i (1 - P(R_i^q{=}1))$,
which has a theoretical maximum of $n$:

| Dataset | $KDA_{cont}$ denominator | Maximum ($n$) | Effective support |
|---|---:|---:|---:|
| SciQ | 40.46 | 884 | **4.58%** |
| OpenBookQA | 86.22 | 500 | **17.24%** |

**Why this matters.** $KDA_{cont}$ was introduced by Moon et al. precisely to avoid
$KDA_{disc}$'s zero-denominator degeneracy. That fix addresses the *catastrophic* case
(denominator exactly 0) but not the *quiet* one: a denominator of 41 still divides, still
returns a number, and still looks like a valid score. The reported 0.9512 for SciQ carries
roughly the statistical weight of a 41-question evaluation, and the metric emits no signal
that this happened.

The original paper anticipated the mechanism in Limitations §6.2 (*too easy questions*) and
§6.3 (*difficulty of measuring PLM ignorance*). Their own Table 11 showed SciQ at 0.96
average with-fact correctness for 2022-era solvers. With a 4B instruct model the
*without*-fact figure has now reached 0.9536 — the ignorance baseline the metric is defined
against has essentially disappeared.

## 6. Finding 2 — probability saturation makes $KDA_{cont}$ redundant

Restricted to four letter logits, the instruct model's softmax saturates almost completely:

| $P(R^q{=}1)$ band | OpenBookQA | SciQ |
|---|---:|---:|
| $[0.90, 1.00]$ | 403 (80.60%) | 842 (95.25%) |
| $[0.75, 0.90)$ | 7 (1.40%) | 0 (0.00%) |
| $[0.50, 0.75)$ | 3 (0.60%) | 1 (0.11%) |
| $[0.25, 0.50)$ | 7 (1.40%) | 1 (0.11%) |
| $[0.00, 0.25)$ | 80 (16.00%) | 40 (4.52%) |
| **Middle bands $[0.25, 0.90)$** | **17 (3.40%)** | **2 (0.23%)** |

The distribution is effectively **bimodal**: the model is either certain or certainly
wrong. **777 of 884 SciQ samples (87.9%)** and **257 of 500 OpenBookQA samples (51.4%)**
have $P(R^q{=}1)$ equal to `1.0` in float32; the largest value strictly below 1.0 anywhere
in either dataset is `0.9999998807907104`.

The consequence: $KDA_{cont} \approx KDA_{disc}$ (0.9407 vs 0.9512 on SciQ; 0.5956 vs
0.5862 on OpenBookQA — differences of ~0.01). The continuous relaxation was designed to
recover graded information from soft probabilities. On a modern instruct LLM there is no
graded information left to recover, and $KDA_{cont}$ degenerates into an expensive
re-derivation of $KDA_{disc}$.

> **Is this an artefact of letter-logit scoring?** Partly. Restricting to four tokens
> discards mass the model assigns elsewhere and sharpens the resulting distribution. But
> the effect is not the cause of the collapse: the *discrete* metric, which does not depend
> on calibration at all, loses 95.4% of SciQ on argmax alone. Saturation makes $KDA_{cont}$
> uninformative; it does not create the support problem. See §9.

## 7. Finding 3 — comparison with the `KDA_small` encoder baseline

Against the four-model, 355.1M-parameter `KDA_small` ensemble from
[`kda_reproduction_summary.md`](kda_reproduction_summary.md), scored on the **same files**:

| | SciQ $Acc_{wof}$ | SciQ $Acc_{wf}$ | OBQA $Acc_{wof}$ | OBQA $Acc_{wf}$ |
|---|---:|---:|---:|---:|
| `KDA_small` (ensemble average) | 38.43% | 70.73% | 32.35% | 39.55% |
| `KDA_small` (pooled probability vote) | 45.48% | 88.57% | 34.80% | 40.60% |
| **Qwen3-4B-Instruct-2507 (4-bit)** | **95.36%** | **99.77%** | **82.60%** | **92.00%** |
| **Change vs. pooled vote** | **+49.88 pp** | **+11.20 pp** | **+47.80 pp** | **+51.40 pp** |

Two things move at once, and they move in opposite directions for the metric's health:

* **The without-fact accuracy roughly doubles.** This is the parametric prior, and it is
  what destroys the metric's support. On SciQ it goes from "the ensemble gets fewer than
  half right" to "the model gets 19 in 20 right" — before seeing any course material.
* **The with-fact accuracy hits the ceiling.** SciQ reaches 99.77% (2 errors in 884), so
  the numerator saturates too. Both terms of the ratio are pinned near their maxima and the
  metric loses discriminative power in both directions simultaneously.

The accuracy *gain* from the fact shrinks correspondingly: the `KDA_small` pooled vote
gained **+43.10 pp** on SciQ from the passage; Qwen3-4B gains **+4.41 pp**. Not because the
passage got less useful, but because there was almost nothing left for it to fix.

Note the reversal in dataset ordering. For the encoders, OpenBookQA was the *harder*
dataset and produced the *lower* KDA (0.2712 vs 0.4715), because a one-clause `fact1`
demands deduction the encoders could not do. For Qwen3-4B, OpenBookQA produces the lower
KDA (0.5862 vs 0.9512) for the **opposite** reason: it retains more usable support (17.4%
vs 4.6%), so it is the dataset on which the metric is still partially alive.

## 8. Error analysis and qualitative case studies

### 8.1. Parametric memorisation — the `both_correct` bucket

843 SciQ and 409 OpenBookQA questions land here. These are the questions the metric cannot
see. Representative cases, all with $P(R^q{=}1) = 1.0000$ and $P(R^{q+f}{=}1) = 1.0000$:

| Dataset | Question | Gold | Target fact |
|---|---|---|---|
| SciQ | *"Vertebrata are characterized by the presence of what?"* | `backbone` | "Vertebrata are characterized by the presence of a backbone…" |
| SciQ | *"What term in biotechnology means a genetically exact copy of an organism?"* | `clone` | "…could a clone, a genetically exact copy of an organism, be developed…" |
| SciQ | *"What is the height above or below sea level called?"* | `elevation` | (paragraph about mountain ranges and elevation) |
| OBQA | *"Predators eat"* | `bunnies` | "predators eat prey" |
| OBQA | *"As the rain forest is deforested the atmosphere will increase with"* | `carbon` | "as the population of plants decreases, carbon in the atmosphere will increase" |

The OpenBookQA *"Predators eat"* item is the paper's own **Figure 1 "bad question (too
easy)"** example — the canonical illustration of a question that fails the KDA criterion.
Qwen3-4B answers it with probability 1.0 without the fact, exactly as the figure predicts a
student would. The metric assigns it weight zero and moves on, which is the correct
behaviour for *that one question* but becomes the failure mode when it applies to 95% of
the dataset.

The SciQ cases show a second mechanism beyond memorisation: **the question stem is a
near-paraphrase of the fact sentence**, so the answer is recoverable from the stem's own
wording plus vocabulary knowledge. "Vertebrata are characterized by the presence of what?"
with `backbone` among the options needs no passage. This is the paper's "High KDA, Low
Likert" failure class (their Table 14) surfacing as a *support* problem rather than a
*scoring* problem.

### 8.2. Strictly fact-dependent — the `wrong_to_correct` bucket

39 SciQ and 51 OpenBookQA questions. These are the questions the metric is actually built
to reward, and the model's behaviour on them is clean — a decisive flip from one option to
another, $P(R^q{=}1) = 0.0000 \rightarrow P(R^{q+f}{=}1) = 1.0000$:

| Dataset | Question | Predicted → Gold | Why the fact was necessary |
|---|---|---|---|
| SciQ | *"About what percentage of the earth's water is fresh water?"* | `two percent` → `three percent` | An arbitrary numeric value; the prior guessed the plausible-sounding "two percent" |
| SciQ | *"What kind of a process is corrosion?"* | `oxidation` → `galvanic` | `oxidation` is the textbook-adjacent distractor; only the passage licenses `galvanic` |
| SciQ | *"What type of mirror is shaped like the outside of a bowl?"* | `concave` → `convex` | A sign convention that must be read, not reasoned |
| OBQA | *"Which characteristic did a person inherit?"* | `length of hair` → `number of nails` | Needs `fact1` "the number of body parts… is an inherited characteristic" |
| OBQA | *"What may have been formed by a volcano?"* | `The great lakes` → `Mt. McKinley` | Needs "mountains are formed by volcanoes" to select the mountain |

These are exactly the items a knowledge-dependency metric should score highly, and they are
the 4.6%/17.4% that still carry signal. The problem is not that the metric misjudges them —
it is that they are all that remains.

### 8.3. The fact actively hurt — the `correct_to_wrong` bucket

Zero cases on SciQ; **4 on OpenBookQA**. Small in count, but diagnostically important: they
are ClashEval's *context bias* appearing inside the KDA construction.

| Question | Options | Gold | Without fact | With fact | The fact |
|---|---|---|---|---|---|
| *"What would cause a human to grow?"* | light waves / **eating wheat** / photosynthesis / marching | `eating wheat` | `eating wheat` (P=1.000) | `photosynthesis` (P=0.997) | "humans eat crops" |
| *"A recyclable material can be"* | transformed / traded / thrown away / **used more times** | `used more times` | `used more times` (P=1.000) | `transformed` (P=0.991) | "recyclable means a material can be recycled" |
| *"I'm an animal with white fur and a large fluffy tail that lives in arctic regions"* | weasel / **snow fox** / wolf / polar bear | `snow fox` | `snow fox` (P=0.960) | `polar bear` (P=1.000) | "arctic animals live in an arctic environment" |
| *"Kinetic energy can be found in objects that move, such as"* | flower pots on a wagon / cars in a lot / kids sleeping / **skateboards ridden all day** | `skateboards…` | `skateboards…` (P=0.798) | `flower pots on a wagon` (P=0.999) | "motion is a source of kinetic energy in an object" |

In all four the model had the right answer from its prior and was **argued out of it** by a
vague, topically-related fact. The `snow fox` case is the clearest: the fact "arctic animals
live in an arctic environment" contains no discriminating information whatsoever, yet it
shifts the model from `snow fox` (0.960) to `polar bear` (1.000) — the more prototypical
arctic animal. The fact functioned as a *topic prime*, not as evidence.

This is a **failure mode invisible to KDA by construction**: these questions have $r^q = 1$,
so they contribute zero to the denominator and are discarded before their harm is counted.
A metric that measures whether a fact *helps* cannot see a fact that *hurts*.

### 8.4. The fact did not help — the `both_wrong` bucket

2 on SciQ, 36 on OpenBookQA. Inspection shows these split into two distinct causes.

**(a) Genuine reasoning failure.** SciQ id=38, *"How do very massive stars end their
lives?"* — gold `become red supergiants`. Without the fact the model says `explode`
(P=0.867); with the passage (which states "Very massive stars become red supergiants") it
switches to `become super novas` (P=1.000). The passage's later sentences about fusion and
supernovae override its opening clause. A real long-context distraction failure.

**(b) Dataset label idiosyncrasy — the dominant cause on OpenBookQA.** Several `both_wrong`
items have gold labels that contradict standard science, and the model confidently gives the
textbook answer instead:

| Question | Options | **OBQA gold** | Model answers (both phases) |
|---|---|---|---|
| *"all cells use cellular respiration to"* | photosynthesize / **release waste** / perform meiosis / release energy | `release waste` | `release energy` (P=1.000) |
| *"DNA is a vehicle for passing"* | clothes types / school grades / **elbow size** / language and dialect | `elbow size` | `language and dialect` (P=1.000) |
| *"Evaporation of water can lead to"* | waterfalls / **blizzards** / earthquakes / hot springs | `blizzards` | `waterfalls` (P=1.000) |

> **These labels were verified against the raw HuggingFace `allenai/openbookqa`
> `additional` config.** All seven case-study items in this section were checked: the
> `answerKey`, the option text, and this repository's `answer_idx` agree in every case —
> **0 mismatches**. The odd labels are genuine properties of OpenBookQA, not a bug in
> `prepare_openbookqa.py` or in the option shuffling.

`release energy` is the scientifically standard completion of "all cells use cellular
respiration to…", and the model is being marked wrong for giving it. This inflates
`both_wrong`, depresses $Acc_{wf}$, and — because these items have $r^q = 0$ — they *do*
enter the $KDA_{disc}$ denominator while never entering the numerator, **pushing OpenBookQA's
KDA down**. Roughly a third of OpenBookQA's 36 `both_wrong` items show this pattern on
inspection, which means part of the OBQA-vs-SciQ KDA gap is label noise rather than
knowledge dependency.

### 8.5. Summary of the error taxonomy

| Bucket | SciQ | OBQA | Enters $KDA_{disc}$? | What it actually measures |
|---|---:|---:|---|---|
| `both_correct` | 843 | 409 | ❌ silently dropped | **Parametric prior** — the metric's blind spot |
| `wrong_to_correct` | 39 | 51 | ✅ numerator + denominator | Genuine knowledge dependency |
| `both_wrong` | 2 | 36 | ✅ denominator only | Reasoning failure *and* dataset label noise |
| `correct_to_wrong` | 0 | 4 | ❌ silently dropped | **Context bias** — the fact caused harm |

Both of the metric's blind spots are on the $r^q = 1$ side. As solver capability rises, that
side grows, and the fraction of the dataset the metric can see shrinks toward zero.

## 9. Limitations and caveats

* **Single solver, single seed.** One model, $|M| = 1$, one option ordering. The original
  KDA is an ensemble metric; a single-model reading is degenerate by construction (see the
  note in [§2.3 of the README](../../README.md#23-continuous-plm-approximation-kda_cont)).
  The *support-collapse* finding does not depend on ensembling — it follows from
  $Acc_{wof}$ alone — but the specific KDA values would shift with a multi-model ensemble.
* **Letter-logit scoring sharpens the distribution.** Restricting the softmax to four
  tokens discards probability mass and inflates apparent confidence. Per-option sequence
  likelihood would give a smoother $P(R^q{=}1)$ and a less extreme §6. It would **not**
  change §5, which rests on argmax accuracy.
* **No option-order debiasing.** LLMs exhibit position bias in multiple choice. Options are
  shuffled once with a fixed seed at dataset-preparation time and not permuted per sample.
  A cyclic-permutation average would be more robust; it was not run here (4× the compute).
* **No calibration check.** The report treats $P(R^q{=}1)$ as the model's belief. Nothing
  here verifies that a 4-bit quantised model's letter probabilities are calibrated, and NF4
  quantisation itself perturbs logits.
* **OpenBookQA label noise is characterised qualitatively, not quantified.** §8.4 reports a
  pattern observed on inspection of the 36 `both_wrong` items and states it affects roughly
  a third; that share is an eyeball estimate, not an annotated count.
* **`correct_to_wrong` is a 4-sample observation.** The context-bias finding in §8.3 is
  directionally consistent with ClashEval but far too small a sample to quantify.
* **Cross-metric comparison is restricted to accuracies.** As stated in §3, the KDA values
  here (sample-level) and in `kda_reproduction_summary.md` (question-level over an ensemble)
  are different quantities and are never compared numerically in this report.

## 10. Reproduction

Download and checksum-verify the weights:

```bash
uv run --active python code/pre_data/download_models.py --models Qwen/Qwen3-4B-Instruct-2507
```

Run the full evaluation (~8 minutes on an RTX 3050):

```bash
uv run --active python code/ex1_reproduce_KDA/kda_qwen_eval.py --datasets obqa sciq --out results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json --log-file results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_eval.log
```

Quick smoke test:

```bash
uv run --active python code/ex1_reproduce_KDA/kda_qwen_eval.py --datasets sciq --limit 20
```

If VRAM is tighter than 3.68 GiB, add `--double-quant`.

### Artefacts

| Path | Contents |
|---|---|
| [`results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json`](../../results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_results.json) | Per-sample records (full 4-way probability vectors for both phases, predictions, `r_q`, `r_q_plus_f`, prompt token counts, per-sample latency) + the summary block |
| [`results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_eval.log`](../../results/ex1_reproduce_KDA_w_modernLLM/kda_qwen3_4b_eval.log) | Execution log — environment, VRAM, per-sample DEBUG records |
| [`results/ex1_reproduce_KDA_w_modernLLM/model_download_report.json`](../../results/ex1_reproduce_KDA_w_modernLLM/model_download_report.json) | SHA-256 verification of every downloaded weight shard |
| [`code/ex1_reproduce_KDA/kda_qwen_eval.py`](../../code/ex1_reproduce_KDA/kda_qwen_eval.py) | The evaluation script |

---

## Related documents

* [`RUN_QWEN2.5.md`](../../RUN_QWEN2.5.md) — running the same
  evaluation with `Qwen2.5-7B-Instruct` on Colab / Kaggle, to test whether the support
  collapse deepens with scale.
* [`kda_qwen2.5_7b_evaluation_report.md`](kda_qwen2.5_7b_evaluation_report.md) — **added
  2026-09-24**, the completed Qwen2.5-7B run this entry anticipated: a mixed, not uniformly
  worsening, scale effect (SciQ support essentially flat, OBQA support more collapsed at 7B
  than at 4B).
* [`kda_reproduction_summary.md`](kda_reproduction_summary.md) — the `KDA_small` encoder
  baseline compared against in §7.
* [`KDA_Paper_Documentation.md`](../papers/KDA_Paper_Documentation.md) — the metric definition and
  the Limitations §6.2/§6.3 this run empirically confirms.
* [`ClashEval_Paper_Documentation.md`](../papers/ClashEval_Paper_Documentation.md) — prior-vs-context
  arbitration, the frame for §8.3.
* [`../README.md`](../../README.md) — project overview; this run is the RQ1 evidence at
  modern-LLM scale.
