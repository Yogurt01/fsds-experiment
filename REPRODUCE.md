# REPRODUCE — KDA_tiny Ensemble Experiment on SciQ and OpenBookQA

Reproduction guide for the experiment that validates the **KDA (Knowledge Dependent
Answerability)** metric on 50 randomly sampled questions from `allenai/sciq` (`test`
split), using **KDA_tiny**: a lightweight ensemble of |M| = 2 simulated students
(~706MB total) in place of the reference `KDA_small` (4 models, ~3.5GB).

Verified environment: Ubuntu (Linux 7.0.0-30-generic), Python 3.12.3, NVIDIA RTX 3050
4GB. A GPU is optional — the scripts fall back to CPU automatically.

---

## Workspace layout

```
experiment/
├── code/                                all Python scripts
│   ├── utils/                           shared, stage-agnostic helpers
│   │   └── paths.py                     project-root anchored path helpers (no absolute paths)
│   ├── pre_data/                        stage 0 — dataset preparation
│   │   ├── prepare_sciq.py              SciQ       -> KDA input format
│   │   └── prepare_openbookqa.py        OpenBookQA -> KDA input format
│   ├── ex1_reproduce_KDA/               experiment 1 — KDA baseline reproduction
│   │   ├── kda_tiny.py                  the KDA_cont ensemble implementation
│   │   ├── run_experiment.py            baseline KDA runner (any prepared dataset)
│   │   └── categorize_kda_results.py    four-bucket contingency analysis
│   └── ex2_counterfactual/              experiment 2 — counterfactual perturbation
│       ├── counterfactual_passage.py    minimal lexical answer -> distractor rewriting
│       └── run_counterfactual_experiment.py   Setting A/B/C sweep + adjusted KDA
├── datasets/
│   ├── sciq/                 sciq_{train,val,test}_full.json, sciq_all_combined.json, sciq_50.json
│   └── openbookqa/           obqa_{train,val,test}_full.json, obqa_all_combined.json, obqa_50.json
├── results/                  outputs and logs, grouped by experiment ordinal
│   ├── ex1_reproduce_KDA_pipeline/   baseline KDA runs, dataset prep logs
│   │   └── openbookqa/               OpenBookQA baseline outputs
│   ├── ex1_reproduce_KDA_w_modernLLM/  Qwen evaluations + model download reports
│   ├── ex1_category_questions/       basic_category/ and complicated_category/
│   ├── ex2_counterfactual/           counterfactual perturbation experiment
│   ├── ex3_student_simulation/       persona-conditioned simulation runs
│   ├── ex4_free_response/            option-free pilots + validation sheets
│   ├── ex5_failure_analysis/         the RQ1 flagged pool
│   └── ex5_failure_audit/            bucket / overlap / corpus-defect audits
├── docs/                     grouped by source, experiment, or role
│   ├── NEXT_PHASE_HANDOFF.md         living session-to-session handoff
│   ├── papers/                       the three source-paper walkthroughs
│   ├── guides/                       operator runbooks (annotation)
│   ├── synthesis/                    cross-cutting, project-wide summaries
│   ├── ex1_reproduce_KDA/            kda_reproduction_summary.md, Qwen3-4B report
│   ├── ex2_counterfactual/           methodology, OBQA analysis, E1/E2, conventions
│   ├── ex3_student_simulation/       persona simulation report
│   ├── ex4_free_response/            free-response plan + pilot results
│   └── ex5_failure_audit/            taxonomy, provenance, buckets, corpus defect
├── question-score/           the reference implementation (untouched, do not move)
└── .venv/
```

Every script resolves its `--data`, `--out` and `--log-file` arguments relative to the
**project root** (`experiment/`), not to the current working directory, so all commands
below can be run from anywhere. Absolute paths passed on the command line are honoured
as-is.

---

## 0. Requirements

- Python >= 3.9 (verified on 3.12.3)
- Internet access (HuggingFace Hub checkpoints + the SciQ dataset)
- ~8GB free disk space (the torch CUDA wheels dominate this)
- GPU **not required**

---

## 1. Install `uv`

If `uv` is not on the machine yet:

```bash
python3 -m venv ~/.local/uv-bootstrap && ~/.local/uv-bootstrap/bin/pip install -U pip uv && mkdir -p ~/.local/bin && ln -sf ~/.local/uv-bootstrap/bin/uv ~/.local/bin/uv && ln -sf ~/.local/uv-bootstrap/bin/uvx ~/.local/bin/uvx
```

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Check it:

```bash
uv --version
```

> Alternative (Astral's official installer): `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## 2. Get the `question-score` source

```bash
cd ~/Downloads/research/fsds/experiment
```

```bash
git clone https://github.com/riiid/question-score.git
```

---

## 3. Create the virtual environment with `uv`

```bash
cd ~/Downloads/research/fsds/experiment && uv venv .venv --python 3.12
```

```bash
source .venv/bin/activate
```

(Windows: `.venv\Scripts\activate`)

---

## 4. Install dependencies with `uv pip`

```bash
uv pip install torch transformers datasets accelerate
```

```bash
uv pip install protobuf
```

`protobuf` is required only by `google/t5-small-ssm-nq`: its SentencePiece vocabulary has
to be converted at load time, and without it the tokenizer raises
`ImportError: SentencePieceExtractor requires the protobuf library`.

```bash
uv pip install -e ./question-score
```

Versions used for this run: `torch 2.13.0+cu130`, `transformers 5.15.1`,
`datasets 5.0.1`, `accelerate 1.14.0`, `question-score 0.0.7`.

### 4.1. Required compatibility patch

`question-score` targets the 2022-era `transformers`. Line 3 of
`question-score/src/question_score/kda.py` imports `AdamW`, a symbol **removed in
`transformers` v5**, so `import question_score` crashes immediately. `AdamW` is never
used anywhere else in the repository, so deleting it from the import line is enough:

```bash
sed -i 's/^from transformers import AutoModelForMultipleChoice, AutoTokenizer, AdamW$/from transformers import AutoModelForMultipleChoice, AutoTokenizer  # AdamW: unused, removed in transformers v5/' question-score/src/question_score/kda.py
```

Verify the environment:

```bash
python -c "import question_score, torch, transformers; print('question_score OK |', 'torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
```

> The unpatched original is preserved at `question-score/src/question_score/kda.py.orig`.
> `code/ex1_reproduce_KDA/kda_tiny.py` does **not** import `question_score` (it reimplements the scoring logic),
> so this patch is only needed to run the reference `KDA_small` for comparison.

---

## 5. Download and format the SciQ data

`code/pre_data/prepare_sciq.py` has two modes.

### 5.1. Full export (default) — all three splits

```bash
python code/pre_data/prepare_sciq.py
```

This downloads `train`, `validation` and `test`, formats every sample, and writes:

| File | Samples | Size |
|---|---|---|
| `datasets/sciq/sciq_train_full.json` | 10,481 | 8.0MB |
| `datasets/sciq/sciq_val_full.json` | 887 | 684KB |
| `datasets/sciq/sciq_test_full.json` | 884 | 693KB |
| `datasets/sciq/sciq_all_combined.json` | 12,252 | 9.6MB |

Every sample has `passage` (from `support`), `question`, `options` (`correct_answer`
plus the three distractors, shuffled), `answer_idx`, `correct_answer`, `id` (position
within the exported split) and `sciq_index` (row index in the original split).
`datasets/sciq/sciq_all_combined.json` adds a `split` attribute to each sample.

Useful flags:

| Flag | Effect |
|---|---|
| `--splits test validation` | Export only the named splits (default: all three) |
| `--no-combined` | Skip `datasets/sciq/sciq_all_combined.json` |
| `--out-dir DIR` | Write the JSON files somewhere other than the working directory |
| `--seed N` | Seed for option shuffling (default `42`) |
| `--allow-empty-support` | Keep samples whose `support` field is empty |
| `--log-file PATH` | Log somewhere other than `results/ex1_reproduce_KDA_pipeline/prep_sciq.log` |
| `--append-log` | Append to the log instead of overwriting it |

Examples:

```bash
python code/pre_data/prepare_sciq.py --splits test --no-combined --out-dir datasets/sciq_alt
```

```bash
python code/pre_data/prepare_sciq.py --allow-empty-support --seed 7 --log-file results/ex1_reproduce_KDA_pipeline/dataset_prep_seed7.log
```

### 5.2. Fixed-size sample — the 50-question experiment set

```bash
python code/pre_data/prepare_sciq.py --mode sample --n 50 --split test --out datasets/sciq/sciq_50.json
```

This regenerates `datasets/sciq/sciq_50.json`, the subset the reported KDA_tiny results were produced
on. The RNG call order in this mode is deliberately unchanged from the original
single-split script, so the file comes out **byte-identical** (verified: md5
`845a9d4e588f434c7e7dbf48fe30a2dd`) and stays consistent with the existing `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json`.

### 5.3. Empty-support filtering

1,427 of the 13,679 raw samples (10.4%) have an empty `support` field. KDA measures the
difference between a with-fact and a without-fact pass, so an empty passage makes both
passes identical and the score meaningless. These rows are **dropped by default**; pass
`--allow-empty-support` to keep them.

| Split | Raw | Removed (empty `support`) | Final |
|---|---|---|---|
| `train` | 11,679 | 1,198 (10.26%) | 10,481 |
| `validation` | 1,000 | 113 (11.30%) | 887 |
| `test` | 1,000 | 116 (11.60%) | 884 |
| **TOTAL** | **13,679** | **1,427** | **12,252** |

### 5.4. Answer-key balance

`answer_idx` is recomputed after shuffling, so the correct answer is not pinned to one
position. The post-shuffle distribution is close to uniform in every split:

| Split | idx 0 | idx 1 | idx 2 | idx 3 |
|---|---|---|---|---|
| `train` | 2,587 (24.7%) | 2,573 (24.5%) | 2,686 (25.6%) | 2,635 (25.1%) |
| `validation` | 237 (26.7%) | 206 (23.2%) | 222 (25.0%) | 222 (25.0%) |
| `test` | 236 (26.7%) | 204 (23.1%) | 222 (25.1%) | 222 (25.1%) |
| **combined** | **3,060** | **2,983** | **3,130** | **3,079** |

No answer position deviates from 25% by more than 1.7 points, so a model cannot game the
benchmark by favouring one slot.

### 5.5. Dataset preparation log

`code/pre_data/prepare_sciq.py` writes `results/ex1_reproduce_KDA_pipeline/prep_sciq.log` (overwritten each run unless `--append-log`)
and mirrors the same records to stdout: raw sample count per split, number removed for
empty `support`, final saved count, per-split and combined `answer_idx` distributions,
and the output paths. Failures at any stage are logged with a full traceback.

Both modes are deterministic for a fixed `--seed`: re-running produces byte-identical
JSON files (verified with `md5sum -c`).

### 5.6. Known data-quality caveat in SciQ

37 of the 12,252 exported samples contain a duplicated option string. In **11 of them
(all in `train`) the duplicate is the correct answer**, so the question has two correct
options — for example `train #272`, *"Saturn is made mostly of helium and what else?"*,
whose options are `['nitrogen', 'hydrogen', 'hydrogen', 'carbon']`.

This is a defect in the upstream SciQ distractors, not in this preprocessing.
`answer_idx` points at the first occurrence, so such items understate any model's
measured accuracy (picking the second copy counts as wrong) and will slightly depress
KDA. None of these 11 rows are in the `test` split, so they do not affect the existing
`datasets/sciq/sciq_50.json` results. Filter them out before training or evaluating on `train` if it
matters for your use case:

```bash
python -c "import json; d=json.load(open('datasets/sciq/sciq_train_full.json')); c=[x for x in d if x['options'].count(x['correct_answer'])==1]; json.dump(c, open('datasets/sciq/sciq_train_clean.json','w'), ensure_ascii=False, indent=2); print(len(d)-len(c), 'removed;', len(c), 'kept')"
```

---

## 6. Run the experiment

```bash
python code/ex1_reproduce_KDA/run_experiment.py
```

This loads the two-model ensemble, scores all 50 questions, writes `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json`, and
streams the full run to both the console and `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_sample50.log`.

Useful flags:

```bash
python code/ex1_reproduce_KDA/run_experiment.py --models Riiid/kda-distilbert-base-uncased-race Riiid/kda-scibert-uncased-race Riiid/kda-distilroberta-base-race --out results/ex1_reproduce_KDA_pipeline/results_3models.json --log-file results/ex1_reproduce_KDA_pipeline/experiment_3models.log
```

| Flag | Effect |
|---|---|
| `--models A B [C ...]` | Choose the ensemble members (at least two) |
| `--device cpu` \| `cuda` | Force a device (auto-detected when omitted) |
| `--verbose` | Stream the DEBUG-level per-model probability records to the console too |
| `--log-file PATH` | Write the execution log somewhere other than `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_sample50.log` |
| `--append-log` | Append to the log file instead of overwriting it |
| `--allow-single-model` | Deliberately reproduce the degenerate \|M\| = 1 baseline |
| `--models KDA_SMALL` | Preset expanding to the paper's official four-model `KDA_small` suite |
| `--strategy auto` \| `resident` \| `sequential` | `resident` keeps all models in VRAM; `sequential` sweeps the dataset one model at a time, unloading between passes. `auto` (default) picks sequential for three or more models |
| `--progress-every N` | INFO progress bar every N samples. `0` (default) auto-selects: one line per sample for <= 100 samples, otherwise a bar every 5% of the run. Per-sample records always reach the log file at DEBUG level |

Runtime: **3.4s of scoring** for 50 questions, **55.6s for the full 884-question test
split** (0.063s/question) on an RTX 3050. The first run also downloads ~707MB of
checkpoints.

### 6.1. Running on the full dataset files

`code/ex1_reproduce_KDA/run_experiment.py` reads whatever JSON file `--data` points at, as long as it holds a
list of samples with `passage` / `question` / `options` / `answer_idx`. The full-export
files from step 5.1 are directly usable — always pair `--data` with a matching `--out`
and `--log-file` so runs do not overwrite each other:

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_test_full.json --out results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_full.json --log-file results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_full.log
```

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_val_full.json --out results/ex1_reproduce_KDA_pipeline/results_val_full.json --log-file results/ex1_reproduce_KDA_pipeline/experiment_val_full.log
```

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_train_full.json --out results/ex1_reproduce_KDA_pipeline/results_train_full.json --log-file results/ex1_reproduce_KDA_pipeline/experiment_train_full.log
```

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_all_combined.json --out results/ex1_reproduce_KDA_pipeline/results_all_combined.json --log-file results/ex1_reproduce_KDA_pipeline/experiment_all_combined.log
```

**Budget the runtime before launching a large run.** At the measured 0.067s/question for
the two-model ensemble on an RTX 3050:

| Data file | Samples | Estimated scoring time |
|---|---|---|
| `datasets/sciq/sciq_50.json` | 50 | ~3s |
| `datasets/sciq/sciq_test_full.json` | 884 | **55.6s (measured)** |
| `datasets/sciq/sciq_val_full.json` | 887 | ~1 minute |
| `datasets/sciq/sciq_train_full.json` | 10,481 | ~12 minutes |
| `datasets/sciq/sciq_all_combined.json` | 12,252 | ~14 minutes |

CPU-only is roughly an order of magnitude slower. `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json` stores the complete
per-model probability vectors for every sample, so output size scales with the input:
expect roughly 3.2KB per sample (~40MB for the full training split).

Two things to know before a full-split run:

- The console prints one `INFO` line per sample, so redirect it for long runs
  (`... | tee run.out`) or rely on the log file. The `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_sample50.log` `DEBUG` records
  add two lines per sample per model.
- `datasets/sciq/sciq_all_combined.json` mixes all three splits. Per-split metrics are not broken out
  by `code/ex1_reproduce_KDA/run_experiment.py`, but each result record keeps its `split` attribute, so the
  breakdown can be recovered from `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json` afterwards.

---

## 7. Where to find the execution log

`code/ex1_reproduce_KDA/run_experiment.py` configures Python's `logging` module with two handlers:

| Handler | Destination | Level | Contents |
|---|---|---|---|
| `FileHandler` | **`results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_sample50.log`** (project root, next to `code/ex1_reproduce_KDA/run_experiment.py`) | `DEBUG` | Everything below, plus a per-sample `DEBUG` line per model with its full 4-way probability vectors for both passes |
| `StreamHandler` | stdout / console | `INFO` | Environment banner, model loading and timing, per-sample KDA progress, accuracy tables, top/bottom-3 breakdown, errors |

The log file is **overwritten on each run** unless `--append-log` is passed. Every record
is timestamped (`%Y-%m-%d %H:%M:%S`) and tagged with its logger name (`kda_experiment`
or `kda_tiny`). Chatty third-party loggers (`httpx`, `huggingface_hub`, `urllib3`, ...)
are pinned to `WARNING` so the HTTP traffic from the Hub does not bury the experiment's
own records.

The last run produced 249 lines, 102 of them `DEBUG` per-model probability records
(2 models x 50 questions + 2 loading records). Inspect it with:

```bash
grep DEBUG results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_sample50.log | head -20
```

Errors are captured too: dataset/JSON read failures, ensemble initialisation failures and
per-sample scoring failures are all logged with `logger.exception` (full traceback). A
sample that fails to score is recorded in the `failures` list of `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json` and
skipped, so one bad sample cannot abort the run.

---

## 8. Output files

| File | Role |
|---|---|
| `code/pre_data/prepare_sciq.py` | Downloads SciQ, formats and shuffles options; full-split export or fixed-size sampling |
| `code/ex1_reproduce_KDA/kda_tiny.py` | `SimulatedStudent` + `KDATiny` — the ensemble KDA_cont implementation |
| `code/ex1_reproduce_KDA/run_experiment.py` | Scores a dataset file, computes metrics, logs the run |
| `datasets/sciq/sciq_train_full.json` | Full `train` split, 10,481 samples |
| `datasets/sciq/sciq_val_full.json` | Full `validation` split, 887 samples |
| `datasets/sciq/sciq_test_full.json` | Full `test` split, 884 samples |
| `datasets/sciq/sciq_all_combined.json` | All three splits merged, 12,252 samples, each tagged with `split` |
| `datasets/sciq/sciq_50.json` | The 50-question subset the reported results were produced on |
| `results/ex1_reproduce_KDA_pipeline/prep_sciq.log` | Timestamped log of the dataset preparation run |
| `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json` | **Official `KDA_small` baseline:** 4-model, 884 questions (4.1MB) |
| `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_small_sciq_test_full.log` | Execution log of the `KDA_small` run (1.0MB, 4,455 DEBUG lines) |
| `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_full.json` | 2-model run on the same 884 questions (2.7MB) |
| `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_full.log` | Execution log of the 2-model test-split run (579KB) |
| `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json` | Earlier 50-question run (see the note below) |
| `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_sample50.log` | Execution log of the 50-question run |
| `code/pre_data/prepare_openbookqa.py` | Downloads OpenBookQA, joins `fact1`, exports the KDA format |
| `code/utils/paths.py` | Project-root anchored path helpers used by every script |
| `datasets/openbookqa/obqa_train_full.json` | Full OpenBookQA `train` split, 4,957 samples |
| `datasets/openbookqa/obqa_val_full.json` | Full OpenBookQA `validation` split, 500 samples |
| `datasets/openbookqa/obqa_test_full.json` | Full OpenBookQA `test` split, 500 samples |
| `datasets/openbookqa/obqa_all_combined.json` | All three OpenBookQA splits merged, 5,957 samples |
| `datasets/openbookqa/obqa_50.json` | First 50 test samples, for smoke tests |
| `docs/ex1_reproduce_KDA/kda_reproduction_summary.md` | Cross-dataset baseline summary and integrity verification |
| `results/ex1_reproduce_KDA_pipeline/prep_openbookqa.log` | Log of the OpenBookQA preparation run |
| `results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json` | **OpenBookQA `KDA_small` baseline:** 4-model, 500 questions (2.1MB) |
| `results/ex1_reproduce_KDA_pipeline/openbookqa/experiment_kda_small_obqa_test_full.log` | Execution log of the OpenBookQA run |
| `results/ex1_reproduce_KDA_pipeline/openbookqa/console_kda_small_obqa_test_full.txt` | Console transcript of the OpenBookQA run |

> `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json` / `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_sample50.log` were produced with an earlier pairing
> (`kda-distilbert-base-uncased-race` + `kda-bert-base-uncased-race`) on the 50-question
> subset. The current default ensemble swaps in `kda-scibert-uncased-race`, so those two
> files are **not** comparable to `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_full.json` sample-for-sample.

`results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json` layout (all keys in English):

```
summary
  ├─ models, ensemble_size, total_parameters, device
  ├─ mean_kda_tiny, median_kda_tiny, std_kda_tiny, min_kda_tiny, max_kda_tiny
  ├─ accuracy_without_fact, accuracy_with_fact, accuracy_gain   (ensemble average)
  ├─ metrics.per_model[<model>].{accuracy_without_fact, accuracy_with_fact,
  │                              accuracy_gain, mean_p_correct_*, mean_weight,
  │                              mean_kda_tiny}
  ├─ metrics.ensemble_average, metrics.pooled_probability_vote
  ├─ n_zero_denominator, scoring_seconds, total_runtime_seconds, formula, failures
results[]  (one entry per question)
  ├─ passage, question, options, answer_idx, correct_answer, id
  │  (+ sciq_index on SciQ; obqa_index / obqa_id / answer_key on OpenBookQA)
  ├─ kda_score, numerator, denominator, zero_denominator
  ├─ mean_p_correct_without_fact, mean_p_correct_with_fact, mean_probability_gain
  ├─ reason, scoring_seconds
  └─ per_model[<model>].{p_correct_without_fact, p_correct_with_fact, weight,
                         weighted_contribution, probability_gain,
                         predicted_idx_*, is_correct_*, probabilities_*}
```

---

## 9. OpenBookQA

The same pipeline runs unchanged on `allenai/openbookqa`. Only the preparation script
differs, because the target fact has to be joined in from a second config.

### 9.1. Prepare the data

```bash
python code/pre_data/prepare_openbookqa.py
```

Exports all three splits plus the two derived files, and logs to
`results/ex1_reproduce_KDA_pipeline/prep_openbookqa.log`:

| File | Samples |
|---|---:|
| `datasets/openbookqa/obqa_train_full.json` | 4,957 |
| `datasets/openbookqa/obqa_val_full.json` | 500 |
| `datasets/openbookqa/obqa_test_full.json` | 500 |
| `datasets/openbookqa/obqa_all_combined.json` | 5,957 (each sample tagged with `split`) |
| `datasets/openbookqa/obqa_50.json` | 50 (first 50 of the test split, for smoke tests) |

`--splits`, `--no-combined` and `--no-debug-set` narrow the export; the layout mirrors the
SciQ one produced by `code/pre_data/prepare_sciq.py`. `obqa_50.json` is a strict prefix of
`obqa_test_full.json`, so ids and `obqa_index` values line up between the two.

Field mapping:

| KDA field | OpenBookQA source |
|---|---|
| `question` | `question_stem` |
| `passage` | `fact1` — the core science fact, used as the target fact |
| `options` | `choices['text']`, 4 texts in their published A/B/C/D order |
| `correct_answer` | the option whose `choices['label']` equals `answerKey` |
| `answer_idx` | 0-3 position of that option |
| `id` | 0-based rank in the exported file (integer) |
| `obqa_index` | 0-based row number in the upstream split (integer) |
| `obqa_id` | upstream string id (e.g. `"8-343"`), kept for traceability |

`fact1` ships only in the `additional` config while `main` is the canonical benchmark,
so the script loads `main` and joins `fact1` on the shared `id`, checking that the
question, options and answer key agree across the two configs before accepting a row.
All 5,957 rows across the three splits join cleanly — **0 dropped** — and the answer key
is spread over all four positions (test: `{0: 138, 1: 126, 2: 132, 3: 104}`; combined:
`{0: 1642, 1: 1476, 2: 1388, 3: 1451}`).

Options are **not** shuffled by default (`--shuffle-options` is available): unlike SciQ,
where the correct answer is a separate field that would otherwise always land at
index 0, OpenBookQA already distributes the key across the four positions.

### 9.2. Run the `KDA_small` baseline

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/openbookqa/obqa_test_full.json --out results/ex1_reproduce_KDA_pipeline/openbookqa/results_kda_small_obqa_test_full.json --log-file results/ex1_reproduce_KDA_pipeline/openbookqa/experiment_kda_small_obqa_test_full.log --models KDA_SMALL --dataset-name allenai/openbookqa --split test --progress-every 250
```

500 questions x 4 models in **155s** on the RTX 3050 (0.31s per question), sequential
passes, peak VRAM set by the largest single checkpoint.

### 9.3. Results — `KDA_small`, 4 models, 500 questions

| model | Acc_wof | Acc_wf | delta_Acc | mean_KDA |
|---|---:|---:|---:|---:|
| `google/t5-small-ssm-nq` | 23.80% | 29.40% | +5.60% | 0.2888 |
| `Riiid/kda-albert-xlarge-v2-race` | 30.00% | 40.60% | +10.60% | 0.2582 |
| `Riiid/kda-mpnet-base-race` | 43.40% | 52.80% | +9.40% | 0.4355 |
| `Riiid/kda-scibert-uncased-race` | 32.20% | 35.40% | +3.20% | 0.2773 |
| **ENSEMBLE AVERAGE** | **32.35%** | **39.55%** | **+7.20%** | **0.2712** |
| **ENSEMBLE pooled prob. vote** | **34.80%** | **40.60%** | **+5.80%** | **0.2712** |

`mean_KDA` per model is the degenerate |M| = 1 case, where KDA_cont collapses to
`P(R^{q+f} = 1)` (see *Why |M| >= 2 is mandatory*). Only the two ensemble rows carry the
true weighted KDA_cont; both quote the same number because KDA_cont is computed once per
question over all four students.

Score shape: mean **0.2712**, median 0.2466, std 0.1130, min 0.0772, max 0.6719.
0 zero-denominator questions, 0 scoring failures.

### 9.4. OpenBookQA vs SciQ (same 4-model `KDA_small` suite)

| | SciQ `test` (884 q) | OpenBookQA `test` (500 q) |
|---|---:|---:|
| mean KDA_cont | 0.4715 | **0.2712** |
| ensemble-average Acc_wof | 38.43% | 32.35% |
| ensemble-average Acc_wf | 70.73% | 39.55% |
| ensemble-average delta_Acc | +32.30% | +7.20% |
| pooled-vote Acc_wof | 45.48% | 34.80% |
| pooled-vote Acc_wf | 88.57% | 40.60% |
| pooled-vote delta_Acc | +43.10% | +5.80% |

The gap is expected from how the two target facts are built, not from a defect in the
run. SciQ's `support` is a retrieved paragraph that usually *contains* the answer string
verbatim, so Setting B is close to an extraction task and accuracy jumps ~43 points.
OpenBookQA's `fact1` is a single general science sentence (mean length 52 characters)
that must be *combined* with commonsense to reach the answer — e.g. the fact
"using less resources usually causes money to be saved" for the question about how to
save for a vacation. Setting B therefore stays hard, the with-fact probability rises only
modestly, and KDA_cont lands near 0.27. Setting A accuracy is also lower (32% vs 38%,
against a 25% chance floor), confirming the questions are genuinely less answerable from
the students' priors.

---

## Why |M| >= 2 is mandatory

The KDA_cont formulation aggregates over a set of simulated students `M`:

```
KDA_cont(q) = sum_{m in M} (1 - P_m(R^q = 1)) * P_m(R^{q+f} = 1)
              / sum_{m in M} (1 - P_m(R^q = 1))
```

This is a **weighted average of the with-fact success probability, weighted by each
student's prior un-answerability** `(1 - P_m(R^q = 1))`. A student who could already
answer the question without the target fact contributes almost nothing to the score,
which is exactly the behaviour the metric is designed to capture: a question is only
knowledge-dependent if students who *did not* know the fact can answer it *once given*
the fact.

With `|M| = 1` the single weight appears in both the numerator and the denominator and
**cancels exactly**:

```
KDA_cont(q) = (1 - P(R^q=1)) * P(R^{q+f}=1) / (1 - P(R^q=1)) = P(R^{q+f}=1)
```

The metric collapses into plain reading comprehension and stops measuring knowledge
dependency at all. The weighting only carries information once several students disagree
about the prior. `KDATiny.__init__` therefore raises a `ValueError` below two models;
`--allow-single-model` exists only to reproduce that degenerate baseline deliberately.

The effect is large in practice. On the full 884-question test split, DistilBERT alone
would score **0.7692** (exactly its mean `P(R^{q+f}=1)`, by the cancellation above),
while the two-student ensemble reports **0.5894** — because SciBERT contributes the
higher weight (mean 0.7184) together with a much lower with-fact success probability
(0.4251). The same pattern held on the 50-question subset: 0.7412 for DistilBERT alone
versus 0.4937 for the ensemble.

**Safety check.** If `sum_m (1 - P_m(R^q = 1)) = 0` — every student answers correctly
with 100% confidence *without* the fact — the score is set to `0` instead of dividing by
zero, a `WARNING` is logged, and the sample is flagged with `zero_denominator: true`.
This did not trigger on any of the 50 SciQ questions (`n_zero_denominator: 0`).

---

## Model selection

| Role | Checkpoint | Size | Parameters | Suite |
|---|---|---|---|---|
| Student 1 | `Riiid/kda-distilbert-base-uncased-race` | ~268MB | 67.0M | `KDA_FULL` |
| Student 2 | `Riiid/kda-scibert-uncased-race` | ~440MB | 109.9M | **`KDA_SMALL`** |

Both are RACE-fine-tuned multiple-choice checkpoints published by the authors, used as-is
with no retraining. SciBERT is a genuine member of the official `KDA_SMALL` suite
(`kda.py:14-19`); DistilBERT is the smallest checkpoint the authors released and comes
from the wider `KDA_FULL` list. The other two `KDA_SMALL` members are not usable here:
`google/t5-small-ssm-nq` is a seq2seq model scored by a different code path, and
`kda-albert-xlarge-v2-race` is far too large for a 4GB card.

**VRAM footprint (measured on the RTX 3050, 3,951MB total):** 710MB allocated with both
students resident, 780MB peak reserved, leaving 3,077MB free — comfortably inside the
4GB limit.

### The official `KDA_small` suite

Pass `--models KDA_SMALL` to use the paper's own four-model ensemble instead:

| Model | Size | Parameters | Pipeline |
|---|---|---|---|
| `google/t5-small-ssm-nq` | ~308MB | 77.0M | seq2seq (`T5Student`) |
| `Riiid/kda-albert-xlarge-v2-race` | ~235MB | 58.7M | multiple choice |
| `Riiid/kda-mpnet-base-race` | ~438MB | 109.5M | multiple choice (no `token_type_ids`) |
| `Riiid/kda-scibert-uncased-race` | ~440MB | 109.9M | multiple choice |

All four load with no missing, unexpected or mismatched weight keys. Their combined
~1.42GB does not leave comfortable headroom for activations on a 4GB card, so this
ensemble runs with `--strategy sequential` automatically. `protobuf` must be installed
for the T5 tokenizer (step 4).

> **Two checkpoint names in circulation do not exist**; the HuggingFace API returns `401`
> for both:
> - `Riiid/kda-scibert-scivocab-uncased-race` -> the real name is
>   **`Riiid/kda-scibert-uncased-race`**, which is what this experiment uses.
> - `Riiid/kda-albert-base-v2-race` -> only `kda-albert-xlarge-v2-race` and
>   `kda-albert-xxlarge-v2-race` were released.

**Do not substitute a base checkpoint that has not been fine-tuned for multiple choice.**
`AutoModelForMultipleChoice` bolts a randomly initialised classification head onto such a
checkpoint, and the outputs are pure noise. A control run with raw
`distilbert-base-uncased` scored **0.2501 with a standard deviation of 0.0010** — exactly
the 1/4 chance level, unable to distinguish any question from any other. Both checkpoints
above were verified to load with **no missing, unexpected or mismatched weight keys**
(`missing_keys`, `unexpected_keys` and `mismatched_keys` all empty).

---

## Findings — official `KDA_small` baseline (4 models, 884 questions)

The reference ensemble from the paper (`question_score/kda.py:14-19`), run on the full
clean test split:

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_test_full.json --models KDA_SMALL --out results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json --log-file results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_small_sciq_test_full.log
```

`--models KDA_SMALL` expands to `google/t5-small-ssm-nq`,
`Riiid/kda-albert-xlarge-v2-race`, `Riiid/kda-mpnet-base-race` and
`Riiid/kda-scibert-uncased-race` (355.1M parameters, ~1.42GB of weights).

### Headline numbers

| Metric | Value |
|---|---|
| **Mean KDA_cont** | **0.4715** |
| Median | 0.4797 |
| Standard deviation | 0.1425 |
| Min / max | 0.0846 / 0.8419 |
| p10 / p25 / p75 / p90 | 0.2688 / 0.3649 / 0.5780 / 0.6556 |
| Samples scored | 884 (0 failed) |
| Zero-denominator samples | **0** |
| Scoring time | **583.51s** (0.660s/question) on an RTX 3050 |
| Total runtime | 585.01s |
| `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_small_sciq_test_full.json` | 4.1MB |
| `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_small_sciq_test_full.log` | 1.0MB (4,690 lines, 4,455 at DEBUG) |

> **Do not compare 0.4715 against the 0.740 in the project README.** That table reports
> the *correlation* between `KDA_small` and human-annotated KDA, not a mean score. The
> number here is the mean KDA_cont over SciQ questions — a different quantity entirely.

### Accuracy per model

| Model | `accuracy_without_fact` | `accuracy_with_fact` | `accuracy_gain` | mean `P(R^q=1)` -> `P(R^{q+f}=1)` | mean weight |
|---|---|---|---|---|---|
| `google/t5-small-ssm-nq` | 39.82% | 63.01% | +23.19% | 0.3838 -> 0.6063 | 0.6162 |
| `Riiid/kda-albert-xlarge-v2-race` | 26.47% | 62.44% | +35.97% | 0.2509 -> 0.2737 | 0.7491 |
| `Riiid/kda-mpnet-base-race` | 51.36% | 89.71% | **+38.35%** | 0.4094 -> 0.8146 | 0.5906 |
| `Riiid/kda-scibert-uncased-race` | 36.09% | 67.76% | +31.67% | 0.2816 -> 0.4251 | 0.7184 |
| **Ensemble average** | **38.43%** | **70.73%** | **+32.30%** | 0.3314 -> 0.5299 | — |
| Pooled probability vote | 45.48% | 88.57% | +43.10% | — | — |

### Score-shape summary

| Band | Count | Share |
|---|---|---|
| KDA > 0.80 (strongly knowledge-dependent) | 3 | 0.3% |
| KDA < 0.25 (weak knowledge dependency) | 60 | 6.8% |
| Passage *hurt* the students (mean gain < 0) | 75 | 8.5% |

### Comparison against the 2-model runs

| Run | \|M\| | Mean KDA | σ | `Acc_wof` | `Acc_wf` | `ΔAcc` | Scoring time |
|---|---|---|---|---|---|---|---|
| **`KDA_SMALL` (official)** | **4** | **0.4715** | 0.1425 | 38.43% | 70.73% | +32.30% | **583.51s** |
| DistilBERT + SciBERT | 2 | 0.5894 | 0.1645 | 37.33% | 79.19% | +41.86% | 55.64s |
| DistilBERT + BERT-base (50-question subset) | 2 | 0.4937 | 0.1242 | 36.00% | 64.00% | +28.00% | 3.35s |

The official ensemble scores **0.118 lower** than the 2-model configuration on the very
same 884 questions, and it costs **10.5x more compute**. Per-question rankings agree only
moderately between the two: **Spearman ρ = 0.455**. Ensemble composition, not just size,
determines both the level and the ordering of KDA scores.

### What the numbers say

1. **ALBERT-xlarge is severely under-confident and drags the metric down.** Its argmax
   accuracy rises from 26.47% to 62.44% (+35.97 points) with the fact, yet its mean
   probability on the correct option barely moves — **0.2509 -> 0.2737**, a gain of just
   +0.023. It emits a near-uniform distribution over the four options on almost every
   question (per-sample `P(R^q=1)` values cluster at 0.248-0.259, i.e. 1/4). Because it
   also looks maximally ignorant without the fact, it earns the **highest weight of the
   four (0.7491)**, so its flat with-fact probabilities dominate the weighted average.
   This single model explains most of the drop from 0.5894 to 0.4715. KDA_cont is a
   probability-based metric, so a model that ranks well but calibrates badly hurts it in
   a way that accuracy tables do not reveal.

2. **MPNet is the strongest student and the one the metric trusts least.** It reaches
   89.71% with-fact accuracy and the highest mean with-fact probability (0.8146). But it
   also has the highest prior (51.36% without the fact, mean `P(R^q=1)` = 0.4094), so it
   carries the **lowest weight (0.5906)** — exactly the intended behaviour: a student who
   already knows the answer says little about whether the question tests the target fact.

3. **The weighting visibly excludes students who already know the answer.** On `id=587`
   (*"What are catalysts in living things called?"*) T5 answers correctly without the
   fact at `P = 0.993`, giving it a weight of **0.007** — it is effectively removed from
   the average, while the three students who did not know contribute in full. `id=292`
   is even starker: T5's weight is **0.000**. This is the mechanism that a single-model
   ensemble cannot express at all.

4. **Adding models compresses the score range.** Only 3 questions (0.3%) exceed 0.80 with
   four students, against 88 (10.0%) with two, and σ falls from 0.1645 to 0.1425.
   Averaging over more — and more differently-calibrated — students pulls scores toward
   the middle. The share of questions where the passage actively *hurts* rises from 2.7%
   to 8.5%, because a question now only needs to confuse one of four students.

5. **The zero-denominator guard still never fired**, even with a T5 model that reaches
   `P(R^q = 1) = 1.000` on some questions. A single confident student drives its own
   weight to zero but cannot zero the *sum* while the other three remain uncertain — a
   concrete illustration of why the guard is a genuine edge case rather than a routine
   branch.

### Top 3 questions by KDA

| KDA | Question | Why |
|---|---|---|
| 0.8419 | *"What are catalysts in living things called?"* (`enzymes`) | Three students jump to 0.75-0.999 with the fact; T5 already knew it and is down-weighted to 0.007 |
| 0.8213 | *"What is the process of breaking an individual into parts followed by regeneration called?"* (`fragmentation`) | Same pattern — T5's weight is 0.000, the other three rise to 0.70-0.99 |
| 0.8065 | *"What are gases called that absorb heat in the atmosphere?"* (`greenhouse gases`) | Near-chance priors across the board; three of four students exceed 0.97 with the passage |

### Bottom 3 questions by KDA

| KDA | Question | Why |
|---|---|---|
| 0.0846 | *"What is the only animal phyla that does not consist exclusively of invertebrates?"* (`chordates`) | Every student gets **worse** with the passage (T5 0.000 -> 0.004, MPNet 0.148 -> 0.014); the negated phrasing plus a passage that never states the exception defeats all four |
| 0.1058 | *"What once most common bird in north america became extinct in the 1800s?"* (`the passenger piegon`) | The correct option is misspelled in SciQ (*"piegon"*) and a near-duplicate distractor (*"the homing piegon"*) competes with it |
| 0.1301 | *"The full range of wavelengths and frequencies of electromagnetic radiation make up the ____"* | T5 already answers at 0.968 without the fact (weight 0.032); ALBERT drops from 0.375 to 0.246 with it |

### Runtime breakdown (sequential passes)

| Pass | Model | Peak VRAM after load | Time | Per sample |
|---|---|---|---|---|
| 1/4 | `google/t5-small-ssm-nq` | 308MB allocated / 340MB reserved | 31.11s | 0.0352s |
| 2/4 | `Riiid/kda-albert-xlarge-v2-race` | 243MB / 258MB | **463.29s** | **0.5241s** |
| 3/4 | `Riiid/kda-mpnet-base-race` | 447MB / 495MB | 41.37s | 0.0468s |
| 4/4 | `Riiid/kda-scibert-uncased-race` | 449MB / 497MB | 36.42s | 0.0412s |

**ALBERT-xlarge alone accounts for 79% of the total runtime** despite having the fewest
parameters of the four (58.7M). ALBERT shares weights across layers, so a small parameter
count hides a wide 2048-dimensional, 24-layer forward pass — parameter count is a poor
proxy for compute here.

Peak VRAM never exceeded **~500MB reserved** against the card's 3,951MB, because only one
model is resident at a time; after each `unload()` the free memory returned to 3,816MB.
Loading all four at once would have been feasible for weights alone (~1.42GB) but the
sequential strategy keeps headroom for activations and makes the run safe on 4GB.

---

## Findings — 2-model ensemble on the full test split (884 questions)

Command:

```bash
python code/ex1_reproduce_KDA/run_experiment.py --data datasets/sciq/sciq_test_full.json --out results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_full.json --log-file results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_full.log
```

### Headline numbers

| Metric | Value |
|---|---|
| **Mean KDA_cont** | **0.5894** |
| Median | 0.5913 |
| Standard deviation | 0.1645 |
| Min / max | 0.1003 / 0.9665 |
| p10 / p25 / p75 / p90 | 0.3757 / 0.4866 / 0.7100 / 0.7988 |
| Samples scored | 884 (0 failed) |
| Zero-denominator samples | **0** |
| Scoring time | **55.64s** (0.063s/question) on an RTX 3050 |
| Total runtime | 61.43s including model loading |
| `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_full.json` size | **2.7MB** (~3.1KB per sample) |
| `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_full.log` size | 579KB (2,797 lines, 2,675 at DEBUG) |

### Accuracy per model

| Model | `accuracy_without_fact` | `accuracy_with_fact` | `accuracy_gain` |
|---|---|---|---|
| `Riiid/kda-distilbert-base-uncased-race` | 38.57% | 90.61% | **+52.04%** |
| `Riiid/kda-scibert-uncased-race` | 36.09% | 67.76% | +31.67% |
| **Ensemble average** | **37.33%** | **79.19%** | **+41.86%** |
| Pooled probability vote | 40.05% | 92.08% | +52.04% |

Mean `P(R^q = 1)` = 0.2943 -> mean `P(R^{q+f} = 1)` = 0.5971.

| Model | mean `P(R^q=1)` | mean `P(R^{q+f}=1)` | mean weight `1 - P(R^q=1)` |
|---|---|---|---|
| `kda-distilbert-base-uncased-race` | 0.3071 | 0.7692 | 0.6929 |
| `kda-scibert-uncased-race` | 0.2816 | 0.4251 | 0.7184 |

### Score-shape summary

| Band | Count | Share |
|---|---|---|
| KDA > 0.80 (strongly knowledge-dependent) | 88 | 10.0% |
| KDA < 0.25 (weak knowledge dependency) | 27 | 3.1% |
| Passage *hurt* the students (mean gain < 0) | 24 | 2.7% |

### What the numbers say

1. **SciQ is genuinely knowledge-dependent at scale.** Across all 884 questions, mean
   without-fact accuracy is 37.33% — barely above the 25% chance floor — and rises to
   79.19% once the target fact is supplied, a **+41.86 point gain**. Mean
   `P(R^q = 1)` of 0.2943 is almost exactly chance, confirming the precondition KDA
   needs: students cannot answer from background knowledge alone.

2. **The zero-denominator guard never fired.** Not one of the 884 questions had
   `sum_m (1 - P_m(R^q = 1)) = 0`, so every reported score is a true ratio rather than
   the fallback value. The guard is still exercised and verified by a synthetic test.

3. **The two students disagree sharply, and the weighting handles it.** DistilBERT-RACE
   reaches 90.61% with the fact; SciBERT-RACE only 67.76%, and its mean with-fact
   probability is far lower (0.425 vs 0.769) despite the science-domain vocabulary that
   should suit SciQ. Because SciBERT is also slightly worse at guessing without the fact,
   it carries the *higher* weight (0.7184 vs 0.6929) and pulls the ensemble mean down to
   0.5894. This is the weighted average behaving as intended: KDA reports what the
   *population* of simulated students can do, not its best member.

4. **SciBERT is under-confident, not wrong.** Pooling the two probability distributions
   before the argmax gives 92.08% with-fact accuracy — *better* than DistilBERT alone.
   So SciBERT's ranking information is useful even though its confidence is flat and
   poorly calibrated for this prompt format.

5. **The tail identifies defective MCQs, not model failures.** The 27 questions scoring
   below 0.25 are dominated by items where the passage does not support the answer. The
   clearest case is sample `id=153` (SciQ #172): the question asks what a glass
   hydrometer measures (`specific gravity`), but its `support` field is a chapter summary
   about **Mendel's pea plants** — an entirely mismatched passage in the source dataset.
   Others fail because a distractor is also present in the passage, e.g. `id=96`, *"the
   study of how organisms interact with their environment"* (`ecology`), whose passage
   lists `biochemistry` and `biology` alongside it. For all 24 negative-gain items the
   passage actively *lowered* the probability of the correct answer.

### Top 3 questions by KDA

| KDA | Question | Why |
|---|---|---|
| 0.9665 | *"In the microbiology lab, what technique refers to the procedures carried out under sterile conditions?"* | Prior at chance (0.303); the passage states the fact verbatim and both students converge (0.985 / 0.935), gain +0.657 |
| 0.9427 | *"Which organs control the amount of water, ions and other substances in the blood ... in urine?"* | Unguessable without the fact; the passage answers it outright for both students |
| 0.9397 | *"What term describes the variety of life and its processes, including the genetic differences among organisms?"* (`biodiversity`) | The passage opens with the definition almost word-for-word; both students jump to 0.990 / 0.892 |

### Bottom 3 questions by KDA

| KDA | Question | Why |
|---|---|---|
| 0.1003 | *"What do you call the study of how organisms interact with their environment and each other?"* (`ecology`) | The passage enumerates `biochemistry`, `biology` and other distractors alongside the answer, so the fact competes with itself — gain **−0.136** |
| 0.1007 | *"The glass hydrometer ... has been calibrated and labeled to measure what?"* (`specific gravity`) | **Mismatched passage in SciQ**: the `support` field is a summary of Mendel's genetics experiments and has nothing to do with the question |
| 0.1062 | *"Most hydrogen atoms have how many protons?"* (`one`) | The passage discusses hydrogen *isotopes* and proton counts across them, which pulls both students toward the numeric distractors — gain **−0.072** |

---

## Earlier run — 50-question subset (`datasets/sciq/sciq_50.json`)

> Produced with a different pairing (`kda-distilbert-base-uncased-race` +
> `kda-bert-base-uncased-race`) and kept for reference in `results/ex1_reproduce_KDA_pipeline/sciq/results_kda_tiny2_sciq_test_sample50.json`. Not
> sample-for-sample comparable with the full test-split run above.

| Metric | Value |
|---|---|
| Mean KDA_cont | 0.4937 |
| Median / std | 0.5100 / 0.1242 |
| Min / max | 0.2242 / 0.7073 |

| Model | `accuracy_without_fact` | `accuracy_with_fact` | `accuracy_gain` |
|---|---|---|---|
| `Riiid/kda-distilbert-base-uncased-race` | 40.00% | 86.00% | +46.00% |
| `Riiid/kda-bert-base-uncased-race` | 32.00% | 42.00% | +10.00% |
| **Ensemble average** | **36.00%** | **64.00%** | **+28.00%** |
| Pooled probability vote | 44.00% | 86.00% | +42.00% |

The 50-question subset is drawn from the same `test` split, and its ensemble mean (0.4937
with the weaker BERT-base student) sits below the full-split figure of 0.5894 with
SciBERT — consistent with SciBERT being the stronger of the two second students.

---

## Passage truncation

22 of the 1,768 passage encodings in the full test run (884 questions x 2 students, 1.2%)
exceeded the 512-token limit and had their passage tail-truncated; each event is recorded
at `DEBUG` level in `results/ex1_reproduce_KDA_pipeline/sciq/experiment_kda_tiny2_sciq_test_full.log`:

```bash
grep "Truncating passage" experiment_kda_tiny2_sciq_test_full.log
```

The `transformers` console warning *"Token indices sequence length is longer than the
specified maximum sequence length (627 > 512)"* comes from the un-truncated length
*measurement* inside `truncate_passage` and is harmless — the tensors actually fed to the
model are always built with `truncation=True, max_length=512`.

---

## Implementation notes

Differences from the reference implementation in `question-score`:

| | `question_score` (original) | `KDA_tiny` |
|---|---|---|
| Ensemble | fixed lists (`KDA_SMALL` / `KDA_LARGE` / `KDA_FULL`) | any `--models` list; `KDA_SMALL` available as a preset |
| Memory strategy | all models loaded at once | `--strategy sequential` evaluates one model at a time and unloads between passes |
| Passage/question separator | concatenated with no separator | a single space is inserted |
| Padding | always padded to a fixed 512 tokens | dynamic padding to the batch maximum (mathematically equivalent — padded positions are masked out by the attention mask) |
| Over-long passages | split into chunks, highest KDA kept | tail truncation (SciQ `support` fields are short enough that this rarely triggers) |

Everything else is preserved: `get_bert_postfix` (replace `_` with the option, otherwise
append it), softmax over the four multiple-choice logits, and the two-pass
without-fact / with-fact protocol.

### Two scoring pipelines

`make_student()` dispatches on the checkpoint name:

- **`EncoderStudent`** (BERT / ALBERT / MPNet / SciBERT / DistilBERT) — all four options
  are packed into one `AutoModelForMultipleChoice` forward pass of shape
  `(1, 4, seq_len)`, and a softmax over the four logits gives the answer distribution.
  Models whose forward signature has no `token_type_ids` (DistilBERT, RoBERTa, MPNet,
  XLNet) are detected by name and called without it.
- **`T5Student`** (`google/t5-small-ssm-nq`) — T5 has no multiple-choice head, so it
  follows `KDA.infer_t5_model` from the original repository: the source is
  `prompt + question` with the option **not** appended, each option is the decoder
  target, and its score is the negative length-normalised NLL of generating it
  (`NLLLoss(reduction="none", ignore_index=pad_token_id)` over a `LogSoftmax`, summed and
  divided by the target length). A softmax over the four negated losses produces the same
  kind of distribution the encoder students return, so the aggregation code is shared.

Both strategies feed the identical `_aggregate()` method, so the resident and sequential
paths are guaranteed to compute the same KDA_cont values.

### Sequential evaluation

With `--strategy sequential` (automatic for three or more models), each model is loaded,
sweeps the entire dataset while its probability vectors are cached, and is then released:

```python
student.model.to("cpu"); del student.model
gc.collect(); torch.cuda.empty_cache()
```

KDA_cont is aggregated across all models only after the final pass. Peak VRAM is
therefore set by the largest single checkpoint rather than the sum, and the log records
allocated / reserved / free memory before each load and after each unload:

```bash
grep VRAM experiment_kda_small_sciq_test_full.log
```
