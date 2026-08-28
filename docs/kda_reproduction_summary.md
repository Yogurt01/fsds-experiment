# KDA Baseline Reproduction — SciQ vs OpenBookQA

Reproduction summary for the **KDA_cont** (Knowledge Dependent Answerability) metric
evaluated with the official **`KDA_small`** four-model suite on two datasets:
`allenai/sciq` (`test`, 884 questions) and `allenai/openbookqa` (`test`, 500 questions).

Both runs use the same code path (`code/run_experiment.py`), the same ensemble, the same
device and the same metric implementation. The only difference is the input file, so the
numbers below are directly comparable.

| | |
|---|---|
| Metric | `KDA_cont(q) = Σ_m (1 − P_m(R^q=1)) · P_m(R^{q+f}=1) / Σ_m (1 − P_m(R^q=1))` |
| Ensemble | `KDA_small`, \|M\| = 4, 355.1M parameters |
| Setting A | prompt = question + options (target fact withheld) → `P(R^q = 1)` |
| Setting B | prompt = question + options + target fact → `P(R^{q+f} = 1)` |
| Device | CUDA (NVIDIA RTX 3050 4GB), sequential model passes |
| Environment | Python 3.12.3, torch 2.13.0+cu130, Linux 7.0.0-30-generic |

---

## 1. Comparative summary tables

### 1.1. SciQ — `test` split, 884 questions

Target fact = the `support` paragraph.

| Model | Acc_wof | Acc_wf | Δ | Mean KDA |
|---|---:|---:|---:|---:|
| `google/t5-small-ssm-nq` | 39.82% | 63.01% | +23.19% | 0.6063 |
| `Riiid/kda-albert-xlarge-v2-race` | 26.47% | 62.44% | +35.97% | 0.2737 |
| `Riiid/kda-mpnet-base-race` | 51.36% | 89.71% | +38.35% | 0.8146 |
| `Riiid/kda-scibert-uncased-race` | 36.09% | 67.76% | +31.67% | 0.4251 |
| **Ensemble (average of models)** | **38.43%** | **70.73%** | **+32.30%** | **0.4715** |
| **Ensemble (pooled probability vote)** | **45.48%** | **88.57%** | **+43.10%** | **0.4715** |

Score shape: mean 0.4715, median 0.4797, std 0.1425, range 0.0846–0.8419.
0 zero-denominator questions, 0 failures. Scoring time 583.5s (0.660s/question).

### 1.2. OpenBookQA — `test` split, 500 questions

Target fact = `fact1`.

| Model | Acc_wof | Acc_wf | Δ | Mean KDA |
|---|---:|---:|---:|---:|
| `google/t5-small-ssm-nq` | 23.80% | 29.40% | +5.60% | 0.2888 |
| `Riiid/kda-albert-xlarge-v2-race` | 30.00% | 40.60% | +10.60% | 0.2582 |
| `Riiid/kda-mpnet-base-race` | 43.40% | 52.80% | +9.40% | 0.4355 |
| `Riiid/kda-scibert-uncased-race` | 32.20% | 35.40% | +3.20% | 0.2773 |
| **Ensemble (average of models)** | **32.35%** | **39.55%** | **+7.20%** | **0.2712** |
| **Ensemble (pooled probability vote)** | **34.80%** | **40.60%** | **+5.80%** | **0.2712** |

Score shape: mean 0.2712, median 0.2466, std 0.1130, range 0.0772–0.6719.
0 zero-denominator questions, 0 failures. Scoring time 153.7s (0.307s/question).

### 1.3. Head-to-head

| | SciQ (884 q) | OpenBookQA (500 q) | Change |
|---|---:|---:|---:|
| Mean KDA_cont | 0.4715 | 0.2712 | −0.2003 (−42.5%) |
| Ensemble-average Acc_wof | 38.43% | 32.35% | −6.08 pp |
| Ensemble-average Acc_wf | 70.73% | 39.55% | −31.18 pp |
| Ensemble-average Δ | +32.30% | +7.20% | −25.10 pp |
| Pooled-vote Acc_wof | 45.48% | 34.80% | −10.68 pp |
| Pooled-vote Acc_wf | 88.57% | 40.60% | −47.97 pp |
| **Pooled-vote Δ** | **+43.10%** | **+5.80%** | **−37.30 pp** |
| Mean P(R^q=1) → P(R^{q+f}=1) | 0.3314 → 0.5299 | 0.2846 → 0.3149 | gain 0.1985 → 0.0303 |

> **Reading the two ensemble rows.** *Average of models* averages the four per-model
> accuracies; *pooled probability vote* averages the four probability distributions first
> and then takes the argmax, which is why it scores higher. Both rows quote the same mean
> KDA because KDA_cont is computed once per question over all four students — it is not an
> average of per-model scores.
>
> **Reading per-model "Mean KDA".** With \|M\| = 1 the weight `(1 − P(R^q=1))` appears in
> both numerator and denominator and cancels exactly, so a single model's KDA_cont
> degenerates to `P(R^{q+f} = 1)`. The per-model column is reported for comparison; only
> the ensemble rows carry a true weighted KDA_cont.

---

## 2. Cross-dataset empirical analysis

### 2.1. Why SciQ scores Δ ≈ +43.10% / KDA 0.4715 and OpenBookQA only Δ ≈ +5.80% / KDA 0.2712

The gap is a property of **what the two datasets call a "target fact"**, not a defect in
the run. Both were scored by identical code, and neither produced a single failure or
zero-denominator question.

**SciQ `support` is a retrieved paragraph that usually states the answer literally.**
Mean length 478 characters (median 349, max 2,940). The correct answer string appears
**verbatim in the support passage for 820 of 884 questions (92.8%)**. Setting B is
therefore close to a reading-comprehension *extraction* task: the student does not need
to reason, only to locate the span. Typical case (the highest-scoring SciQ question,
KDA 0.8419):

> **Q:** What are catalysts in living things called?
> **Fact:** "… Many of these reactions require catalysts so they will occur quickly enough
> to support life. **Catalysts in living things are called enzymes.** …"
> **Answer:** enzymes

The answer sentence is present word-for-word, so mean P(R^{q+f}=1) climbs to 0.5299 and
the pooled vote reaches 88.57% accuracy.

**OpenBookQA `fact1` is a short, general scientific principle that must be combined with
commonsense.** Mean length 51.9 characters (median 47) — a single clause, not a passage.
The correct answer appears verbatim in only **47 of 500 facts (9.4%)**. Setting B remains
a genuine *deduction* task: the fact supplies one premise, and the student must supply
the second from world knowledge. Representative cases:

| KDA | Question | `fact1` | Answer | Missing premise the model must supply |
|---:|---|---|---|---|
| 0.0772 (min) | What would cause a human to grow? | humans eat crops | eating wheat | *wheat is a crop* |
| 0.2715 (≈median) | Some blind people have demonstrated bat-like skills by: | bats can echolocate | using sound to 'see' | *echolocation = using sound to perceive* |
| 0.6719 (max) | What type of useful product can be made from the moving winds? | wind is used for producing electricity | electricity | *none — the answer is stated verbatim* |

In the median case the fact moves the ensemble probability from 0.284 to 0.279 — it does
not help at all, because the model that cannot make the bats→sound link gains nothing
from being told that bats echolocate. Across the whole split, mean P(R^{q+f}=1) rises
only from 0.2846 to 0.3149, a gain of 0.0303 versus 0.1985 on SciQ.

**The verbatim-containment split confirms the mechanism directly.** Partitioning *each*
dataset by whether the correct answer string occurs in its own target fact isolates the
effect from any other dataset difference:

| Subset | SciQ | OpenBookQA |
|---|---|---|
| Answer verbatim in fact | n = 820 (92.8%), **mean KDA 0.4806**, P: 0.334 → 0.540 | n = 47 (9.4%), **mean KDA 0.3725**, P: 0.309 → 0.423 |
| Answer **not** in fact | n = 64 (7.2%), **mean KDA 0.3548**, P: 0.304 → 0.403 | n = 453 (90.6%), **mean KDA 0.2607**, P: 0.282 → 0.304 |

Within each dataset, verbatim-containing facts score **0.11–0.13 KDA higher** than
deductive ones, and the two datasets' verbatim subsets behave far more alike than either
dataset's two halves do. The headline gap is therefore mostly a *mixture* effect: SciQ is
93% extraction, OpenBookQA is 91% deduction.

**Setting A confirms the questions themselves are harder, not broken.** OpenBookQA's
without-fact accuracy is 32.35% (pooled 34.80%) against a 25% chance floor, versus 38.43%
(pooled 45.48%) for SciQ. The students' priors answer fewer OpenBookQA questions, which
*raises* the KDA weight `(1 − P(R^q=1))` — mean weight 0.62–0.77 per model. A low KDA on
OpenBookQA is therefore not caused by the questions being pre-answerable; the numerator
is small because the fact genuinely fails to make them answerable for these students.

**Interpretation for the metric.** This is the intended behaviour of KDA_cont, not a
failure mode: it measures how much a target fact makes a question answerable *to a
simulated student of this capacity*. A 355M-parameter RACE-tuned ensemble can extract but
largely cannot deduce, so a one-clause premise leaves it near its prior. Any comparison of
absolute KDA values across datasets must therefore control for target-fact style — SciQ
KDA and OpenBookQA KDA are not on a common scale.

### 2.2. Model behaviour breakdown

Per-model KDA, side by side (per-model = degenerate \|M\| = 1 case):

| Model | Type | KDA SciQ | KDA OBQA | Retention | Δ SciQ | Δ OBQA |
|---|---|---:|---:|---:|---:|---:|
| `Riiid/kda-mpnet-base-race` | encoder (MPNet) | **0.8146** | **0.4355** | 53% | +38.35% | +9.40% |
| `google/t5-small-ssm-nq` | seq2seq (T5) | 0.6063 | 0.2888 | 48% | +23.19% | +5.60% |
| `Riiid/kda-scibert-uncased-race` | encoder (SciBERT) | 0.4251 | 0.2773 | 65% | +31.67% | +3.20% |
| `Riiid/kda-albert-xlarge-v2-race` | encoder (ALBERT) | 0.2737 | 0.2582 | 94% | +35.97% | +10.60% |

**`Riiid/kda-mpnet-base-race` is by far the most fact-sensitive student, on both
datasets.** It leads every column that matters: highest Acc_wof (51.36% / 43.40%), highest
Acc_wf (89.71% / 52.80%), highest per-model KDA (0.8146 / 0.4355), and the largest
accuracy gain on SciQ (+38.35%). On OpenBookQA its KDA of 0.4355 is 61% above the
ensemble's 0.2712 — no other student comes close to that separation. Two consequences:

* Its mean weight is the **lowest** of the four (0.5906 on SciQ, 0.6206 on OpenBookQA),
  because `(1 − P(R^q=1))` shrinks as prior accuracy rises. So the single student that
  reads the fact best is also the one the ensemble down-weights most — this is exactly the
  behaviour KDA_cont is designed for, and it is why the ensemble KDA (0.4715 / 0.2712)
  sits well below mpnet's own value rather than being dragged up by it.
* It is the model a single-model (\|M\| = 1) shortcut would most distort. Reporting mpnet
  alone would put SciQ KDA at 0.8146 instead of 0.4715 — a 73% overstatement.

**`google/t5-small-ssm-nq` behaves unlike the encoders and is the clearest
dataset-dependent case.** As a closed-book NQ-finetuned seq2seq model it carries real
world knowledge, giving it the second-highest SciQ Acc_wof (39.82%) — it answers many
SciQ questions from memory before seeing any fact. But it converts facts into answers
least well of any model on SciQ (Δ +23.19%, the smallest gain there), and on OpenBookQA it
collapses to the **lowest Acc_wof of all four (23.80%, essentially the 25% chance floor)**
with a Δ of only +5.60%. Its KDA retention across datasets is the second-worst (48%).
The reason is architectural: the encoders score all four options jointly through a
multiple-choice head, whereas T5 is scored by per-option sequence likelihood, which
degrades when the "passage" is a one-clause premise rather than a paragraph containing
the answer span. T5 is the student that most needs an extractive passage.

**The encoders split into two regimes.** SciBERT sits mid-field on SciQ (KDA 0.4251) but
has the *smallest* OpenBookQA gain of all four models (+3.20%) — its science-vocabulary
pretraining helps with SciQ's technical passages and gives it almost nothing on
OpenBookQA's plain-language commonsense premises. ALBERT-xlarge is the mirror image: the
weakest prior (Acc_wof 26.47% on SciQ, near chance) and the lowest SciQ KDA (0.2737), yet
it is the **most stable across datasets (94% retention)** and posts the largest OpenBookQA
accuracy gain of any model (+10.60%). Because its prior is weak its KDA weight is the
highest (0.7491 / 0.7479), so it contributes most to the ensemble numerator — ALBERT and
mpnet pull the ensemble in opposite directions, which is precisely why \|M\| ≥ 2 is
mandatory for a non-degenerate score.

---

## 3. Reproduction integrity verification

### 3.1. Bit-for-bit reproduction of predictions — **verified**

The OpenBookQA baseline was executed twice, end to end, from the same input file with the
same command. Comparing the committed results file against the independent re-run:

| Check | Result |
|---|---|
| Samples scored | 500 vs 500 |
| All 4,000 probability vectors (4 models × 500 questions × 2 settings) | **identical** |
| All argmax predictions (`predicted_idx_without_fact` / `_with_fact`) | **identical** |
| All 500 `kda_score` values | **identical at full float precision** |
| Per-sample records (every field except the per-sample timing) | **identical** |
| Summary block (every field except wall-clock runtime) | **identical** |
| `mean_kda_tiny` | `0.27116560020399944` vs `0.27116560020399944` |

Only wall-clock timings differ between runs (153.65s vs the re-run), as expected. Scoring
is deterministic: inference runs under `torch.no_grad()` with no sampling, no dropout and
no data shuffling, and the option order is fixed in the prepared dataset file.

Dataset preparation is likewise deterministic and **non-destructive**: re-exporting all
three OpenBookQA splits reproduced `obqa_test_full.json` byte-for-byte (MD5 verified
against the copy the baseline was scored on), so the published results still correspond
exactly to the file on disk.

### 3.2. Artefact locations

All paths are relative to the project root (`experiment/`). No absolute path is hardcoded
anywhere: `code/paths.py` derives `PROJECT_ROOT` from its own file location, and every
script resolves `--data`, `--out` and `--log-file` against it, so commands run identically
from any working directory.

**Code**

| Path | Role |
|---|---|
| `code/paths.py` | `PROJECT_ROOT` / `DATASETS_DIR` / `RESULTS_DIR` + `resolve()`, `ensure_parent()` |
| `code/kda_tiny.py` | `KDATiny` ensemble, the `KDA_SMALL` preset, the KDA_cont implementation |
| `code/prepare_sciq.py` | SciQ → KDA input format |
| `code/prepare_openbookqa.py` | OpenBookQA → KDA input format (joins `fact1` from the `additional` config) |
| `code/run_experiment.py` | Baseline runner used for both datasets |

**Datasets**

| Path | Samples |
|---|---:|
| `datasets/sciq/sciq_test_full.json` | 884 |
| `datasets/openbookqa/obqa_train_full.json` | 4,957 |
| `datasets/openbookqa/obqa_val_full.json` | 500 |
| `datasets/openbookqa/obqa_test_full.json` | 500 |
| `datasets/openbookqa/obqa_all_combined.json` | 5,957 (each sample tagged with `split`) |
| `datasets/openbookqa/obqa_50.json` | 50 (strict prefix of the test split, for smoke tests) |

All OpenBookQA splits joined cleanly: **0 rows dropped** out of 5,957, with question,
options and answer key verified to agree between the `main` and `additional` configs on
every row.

**Results and logs**

| Path | Content |
|---|---|
| `results/results_kda_small_test_full.json` | SciQ baseline, full per-sample records (4.1MB) |
| `results/experiment_kda_small_test_full.log` | SciQ execution log (1.0MB, DEBUG per sample per model) |
| `results/openbookqa/results_kda_small_obqa_test_full.json` | OpenBookQA baseline, full per-sample records (2.1MB) |
| `results/openbookqa/experiment_kda_small_obqa_test_full.log` | OpenBookQA execution log (567KB) |
| `results/openbookqa/console_kda_small_obqa_test_full.txt` | OpenBookQA console transcript |
| `results/openbookqa_prep.log` | OpenBookQA preparation log (per-split counts, drop reasons) |
| `results/dataset_prep.log` | SciQ preparation log |

Each log records the Python/torch/platform versions, the resolved absolute input and
output paths, the ensemble membership, per-model VRAM before and after load, and a DEBUG
line per question per model carrying both full 4-way probability vectors — enough to
recompute every number in this document without re-running any model.

### 3.3. Commands

```bash
python code/prepare_openbookqa.py
```

```bash
python code/run_experiment.py --data datasets/openbookqa/obqa_test_full.json --out results/openbookqa/results_kda_small_obqa_test_full.json --log-file results/openbookqa/experiment_kda_small_obqa_test_full.log --models KDA_SMALL --dataset-name allenai/openbookqa --split test --progress-every 250
```

```bash
python code/run_experiment.py --data datasets/sciq/sciq_test_full.json --out results/results_kda_small_test_full.json --log-file results/experiment_kda_small_test_full.log --models KDA_SMALL --progress-every 250
```

### 3.4. Checksums (MD5)

```
897c48caf7c999433cb7e9acb366896c  datasets/openbookqa/obqa_50.json
39f8f7b4608069b876be7ff098b3c860  datasets/openbookqa/obqa_all_combined.json
f853c4deb8c4b7f496c24767fd8e02a5  datasets/openbookqa/obqa_test_full.json
7a83b5f213e55200022c851dd52a282f  datasets/openbookqa/obqa_train_full.json
db0ed83284cfcfade93756c523cf56a5  datasets/openbookqa/obqa_val_full.json
99a9c7c645e79bc603cb1b958d6af97b  datasets/sciq/sciq_test_full.json
8afde75a727651fe2a28df92559d69f6  results/openbookqa/results_kda_small_obqa_test_full.json
9ec4d58239c6d4b867a17867469bdd66  results/results_kda_small_test_full.json
```

### 3.5. Known naming caveat

The `KDA_small` suite is published under the HuggingFace org **`Riiid`** (three i's), not
`Riid`. Two names cited in the original paper do not resolve on the Hub and are not used
here: `Riiid/kda-scibert-scivocab-uncased-race` (the real name is
`Riiid/kda-scibert-uncased-race`) and `Riiid/kda-albert-base-v2-race` (only the xlarge and
xxlarge ALBERT variants were released).

---

*See `REPRODUCE.md` at the project root for the full environment setup, dataset
preparation options, and the per-question findings for each run.*
