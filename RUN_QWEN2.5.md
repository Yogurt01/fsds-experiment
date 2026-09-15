# `RUN_QWEN2.5.md` — Cloud runbook for Qwen2.5-7B counterparts

**Living checklist and execution guide in one file.** Every round that completes new
Qwen3-4B-dependent work appends an entry here. Do not treat it as finished.

**Why this file exists.** The local machine is an RTX 3050 Laptop (4 GB).
`Qwen3-4B-Instruct-2507` fits under 4-bit NF4 (2.66 GB weights, 3.41 GB peak).
`Qwen2.5-7B-Instruct` (7.62B params) does not fit with useful headroom, so every Qwen2.5
counterpart runs on Colab/Kaggle instead.

**Portability.** §1 is the setup every entry needs; each entry section is then self-contained
from there. Where a repository script is required, §1.5 shows how to get it into the notebook —
and Entry 1's script is reproduced inline in [Appendix A](#appendix-a--entry-1s-script-inline)
so it can be run with no repository access at all.

> **Merged 2026-09-11.** This file previously had a companion,
> `docs/guides/RUN_QWEN2.5_7B_CLOUD_GUIDE.md`, holding the platform/environment setup, the
> step-by-step walkthroughs and the troubleshooting table for Entry 1. The two had drifted —
> three different install cells and a contradictory precision recommendation — so they were
> merged here and the companion deleted. §1 is the reconciled shared setup; the Entry-1-specific
> walkthrough now lives in §2.

---
## Contents

| § | | |
|---|---|---|
| [0](#0-status-board) | **Status board** | what is queued, blocked, and on the critical path |
| [0.2](#02-how-to-use-this-file) | How to use this file | the two reading paths |
| [1](#1-common-setup--every-entry) | **Common setup — every entry** | platform · install · token · **precision policy** · files in/out |
| [2](#2-entry-1--kda-saturation-evaluation) | Entry 1 — KDA saturation evaluation | + Colab/Kaggle walkthrough, datasets, outputs |
| [3](#3-entry-2--persona-simulation-sciq--obqa) | Entry 2 — Persona simulation | |
| [4](#4-entry-3--free-response-second-judge--critical-path) | Entry 3 — Free-response second judge | ⚠ **critical path** |
| [5](#5-entry-4--free-response-generation-optional-not-blocking) | Entry 4 — Free-response generation | optional |
| [6](#6-troubleshooting) | Troubleshooting | applies to every entry |
| [7](#7-conventions-for-returned-files) | Conventions for returned files | naming + summary-block discipline |
| [8](#8-changelog) | Changelog | |
| [A](#appendix-a--entry-1s-script-inline) | **Appendix A** — Entry 1's script, inline | for zero-repository-access use |

---

## 0. Status board

| # | Task | Qwen3-4B status | Qwen2.5-7B status | Blocking? |
|---|---|---|---|---|
| 1 | KDA saturation evaluation | ✅ done — `docs/ex1_reproduce_KDA/kda_qwen3_4b_evaluation_report.md` | ⬜ **not run** | no |
| 2 | Persona simulation (SciQ + OBQA) | ✅ done — `docs/ex3_student_simulation/student_persona_simulation_report.md` | ⬜ **not run** | no |
| 3 | Free-response **second judge** | ✅ Qwen3-4B is judge — pilot done | ⬜ **not run** | **YES — blocks the full free-response run** |
| 4 | Free-response **generation** (cells A′/B′) | ✅ pilot done (n=25) | ⬜ not run (optional) | no |

**Only entry 3 is on the critical path.** The full free-response run is blocked on
(a) the human validation gate (κ ≥ 0.70, see `code/ex4_free_response/compute_kappa.py`) and
(b) entry 3 below.

> **Gate (a) cleared 2026-09-08.** All 74 rows of `validation_sheet_pilot_bidir.csv` were
> hand-labelled and scored: κ = **0.754** on the both-directions-agree verdict (0.614 forward-only),
> so `gate_passed = true`. Report: `results/ex4_free_response/kappa_report.json`; detail and caveats
> in `docs/NEXT_PHASE_HANDOFF.md` §4.0. **Entry 3 remains outstanding and is independently
> required** — clearing (a) does not release the full run.

### 0.1 Exhaustiveness audit

Every place an LLM is used anywhere in this project, verified by
`grep -rln "Qwen" code/` plus a check of which stages load a model at all:

| Script | Model | Needs a Qwen2.5 counterpart? |
|---|---|---|
| `code/ex1_reproduce_KDA/kda_qwen_eval.py` | Qwen3-4B | yes → entry 1 |
| `code/ex3_student_simulation/run_student_simulation.py` | Qwen3-4B | yes → entry 2 |
| `code/ex4_free_response/run_free_response.py` | Qwen3-4B (generator **and** judge) | yes → entries 3 (judge) and 4 (generator) |
| `code/ex1_reproduce_KDA/run_qwen7b_eval.py` | Qwen2.5-7B | it *is* the entry-1 script |
| `code/pre_data/download_models.py` | both | download only, no inference |
| `code/ex1_reproduce_KDA/kda_tiny.py`, `run_experiment.py` | KDA_small encoders (30–110M) | no — they run locally |
| `code/ex2_counterfactual/*` | **none** — pure regex | no |
| `code/ex1_reproduce_KDA/categorize_kda_results.py`, `code/ex5_failure_audit/*` | **none** — pure computation | no |

Nothing else in the pipeline invokes a language model.

---

### 0.2 How to use this file

There are two reading paths, and they barely overlap:

| You want to… | Read |
|---|---|
| **Check what is queued or blocked** | §0's status board, then the changelog (§8). Stop there. |
| **Actually run an entry in a notebook** | §1 top to bottom *once* per notebook, then the one entry section you need (§2–§5) and nothing else. §6 if something breaks. |

Each entry section is contiguous: once you are in it, you should not need to jump back out
except to §6 (troubleshooting) or §7 (file-naming conventions for what you send back).

---

## 1. Common setup — every entry

Run this once per notebook, whichever entry you are here for. Everything in §1 applies to all
four entries; the entry sections assume it has been done.

### 1.1 Choose your platform and enable the GPU

`Qwen2.5-7B-Instruct` has **7.62B parameters**. Memory needed for weights alone:

| Precision | Weights | Fits on |
|---|---:|---|
| `bf16` / `fp16` | ~15.2 GB | A100 40GB ✅ · single T4 16GB ❌ (no room for activations) · 2×T4 sharded ✅ |
| `8bit` (LLM.int8) | ~8.0 GB | single T4 16GB ✅ |
| `4bit` (NF4) | ~4.5 GB | single T4 16GB ✅ (lots of headroom) |

Recommended configuration per platform:

| Platform | GPU | VRAM | `--precision` | `--batch-size` | Est. runtime (1,384 samples) |
|---|---|---:|---|---:|---|
| **Colab free** | 1× T4 | 16 GB | `8bit` | 8 | ~35–60 min |
| **Colab free** (faster) | 1× T4 | 16 GB | `4bit` | 16 | ~20–35 min |
| **Colab Pro** | 1× L4 | 24 GB | `fp16` | 16 | ~12–20 min |
| **Colab Pro+** | 1× A100 | 40 GB | `bf16` | 32 | ~5–10 min |
| **Kaggle** | 2× T4 | 2×16 GB | `fp16` | 16 | ~15–25 min |

> **T4 note.** The T4 is compute capability 7.5 and has **no bfloat16 support**. Use
> `fp16`, `8bit`, or `4bit` on T4 — never `bf16`. The script's default
> `--precision auto` detects this and picks for you.
>
> **8-bit is slower than 4-bit.** LLM.int8() runs mixed-precision decomposition on
> outlier channels, which costs more time than NF4 dequantisation. If you only care about
> throughput on a free T4, `4bit` is both smaller *and* faster; `8bit` is marginally more
> faithful to full precision.
>
> **The `--precision` column above is what each platform *can* run, not what you *should* run.**
> Per §1.4 the answer is `4bit` on every platform, because that is what the local Qwen3-4B run
> used. The other columns are there for the case where comparability is not the goal.

**Enable the GPU before anything else.** Colab: `Runtime → Change runtime type → T4 GPU`.
Kaggle: right-hand panel → `Accelerator → GPU T4 x2`.

Verify:

```python
!nvidia-smi
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("bf16 supported:", torch.cuda.is_bf16_supported())
```

---

### 1.2 Install

One canonical cell. (Three different variants of this cell used to exist across the two
merged files; this is their union, keeping the `transformers` pin and the Kaggle numpy fix.)

```python
!pip install -q -U "transformers>=4.44" accelerate bitsandbytes datasets pandas huggingface_hub
import torch, transformers
print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))
```

On **Kaggle**, pin `numpy` first if you hit a binary-compatibility error on import:

```python
!pip install -q -U "numpy<3" "transformers>=4.44" accelerate bitsandbytes datasets pandas huggingface_hub
```

Confirm what actually loaded — Colab pre-installs older copies, and a restart is sometimes
required after upgrading:

```python
import transformers, accelerate, bitsandbytes, datasets
print("transformers ", transformers.__version__)
print("accelerate   ", accelerate.__version__)
print("bitsandbytes ", bitsandbytes.__version__)
print("datasets     ", datasets.__version__)
```

> If `bitsandbytes` raises `CUDA Setup failed`, restart the runtime
> (`Runtime → Restart session`) and re-run this cell. The wheel needs to be imported against
> the CUDA runtime that is loaded at process start.

### 1.3 `HF_TOKEN` (optional)

`Qwen/Qwen2.5-7B-Instruct` is a **public** repository, so a token is *not required*. Set
one anyway for higher rate limits and faster downloads.

**Google Colab** — click the 🔑 key icon in the left sidebar, add a secret named
`HF_TOKEN`, toggle "Notebook access" on, then:

```python
import os
from google.colab import userdata
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN loaded from Colab secrets")
except Exception as exc:
    print("No HF_TOKEN set — continuing unauthenticated:", exc)
```

**Kaggle** — `Add-ons → Secrets → Add a new secret` named `HF_TOKEN`, attach it to the
notebook, then:

```python
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception as exc:
    print("No HF_TOKEN set — continuing unauthenticated:", exc)
```

> Never paste a token literal into a notebook cell you intend to share — notebook outputs
> and revision history persist it. Use the secrets manager.

---

### 1.4 Load the model — and the precision policy

> ### ⚠ Precision is a methodological choice, not a performance knob
>
> **Default to 4-bit NF4.** The local Qwen3-4B run these entries are compared against used
> 4-bit NF4 with fp16 compute. Matching it means quantisation is *not* a confound when you diff
> the two models — which is the entire point of every entry here.
>
> `8bit` and `fp16`/`bf16` will run faster or marginally more faithfully, but **they forfeit
> comparability with the local run.** If you use them, the numbers are still internally valid,
> but any Qwen2.5-vs-Qwen3 delta becomes confounded with the precision change and must be
> reported as such. The two merged source files used to disagree about this — the status board
> mandated NF4 for comparability while the guide's paste-ready commands used `8bit` and `fp16`.
> NF4 is the policy.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=quant, dtype=torch.float16, device_map={"": 0}
).eval()
print(f"VRAM after load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

Entry 1's script (§2) takes `--precision` and defaults to `auto`; **pass `--precision 4bit`
explicitly** if you intend to compare against the local run.

### 1.5 Getting files into the notebook

Entries 1 and 3 are fully inline — nothing to upload. Entry 2 needs the repo:

```python
# Option A: upload a zip of the repo through the Files pane, then
!unzip -q experiment.zip -d /content/ && ls /content/experiment/code

# Option B: mount Drive
from google.colab import drive; drive.mount('/content/drive')
```

Datasets needed (upload alongside, preserving relative paths):
`datasets/sciq/sciq_test_full.json` (884 items) · `datasets/openbookqa/obqa_test_full.json` (500).

---

### 1.6 Getting results back out

Applies to every entry. The per-entry sections say *which* files to expect; this is the mechanism.

**Colab:**

```python
import shutil
from google.colab import files
shutil.make_archive("/content/out", "zip", "/content/out")
files.download("/content/out.zip")
```

**Kaggle:** anything written under `/kaggle/working/` appears in the notebook's **Output** panel — click the download icon, or `Save Version` and grab the artefacts from the version page.

Naming and summary-block discipline for what you send back: §7.

---

## 2. Entry 1 — KDA saturation evaluation

**What it establishes.** Whether parametric saturation is specific to Qwen3-4B or general to
modern instruct models. Qwen3-4B reaches `Acc_wof` = **0.954** on SciQ / **0.826** on OBQA, which
is the motivating fact for the whole programme. If Qwen2.5-7B also saturates, the finding
generalises; if it does not, the local conclusion is model-specific.

**What to run.** The standalone script already exists and imports nothing from the repo:
[`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](code/ex1_reproduce_KDA/run_qwen7b_eval.py). Upload
that one file.

```bash
# installs: §1.2
python run_qwen7b_eval.py --datasets sciq obqa --precision 4bit --out-dir ./kda_out
```

**Runtime / VRAM:** ~25–40 min for 1,384 items at 4-bit on a T4; ~5 GB VRAM.

**Expected outputs — download and place at:**

```
results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_results.json      (parallel to kda_qwen3_4b_results.json)
results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_eval.log
docs/ex1_reproduce_KDA/kda_qwen2.5_7b_evaluation_report.md                            (parallel to kda_qwen3_4b_evaluation_report.md)
```

**Comparison analysis once both exist:**

1. **Saturation diff** — `accuracy_without_fact` per dataset, Qwen2.5-7B minus Qwen3-4B. The
   headline question: does a larger model saturate more?
2. **KDA_disc denominator diff** — `kda_disc_denominator` is the count of items answered wrong
   without the fact. It is the metric's usable population; Qwen3-4B leaves only 41 on SciQ. If
   Qwen2.5-7B leaves fewer, saturation is worsening with scale and KDA's dynamic range is
   shrinking for structural reasons.
3. **Bucket diff** — `both_wrong` and `correct_to_wrong` counts. Qwen3-4B leaves 2 and 0 on SciQ.
   Feed the new file into `code/ex5_failure_audit/bucket_diagnostics.py` (add a `qwen25` entry to
   `DATASET_SPECS`/`confidence_gate_qwen`) to get a third row in the confidence-gate table.
4. **Item-level agreement** — join on `id` and report the fraction where both models are correct
   without the fact. High agreement means the two share a memorisation profile.

---

### 2.1 Step-by-step: Google Colab

1. **New notebook** → `Runtime → Change runtime type` → **T4 GPU** (or A100 on Pro+) → Save.
2. Run the **§1.2 install** cell. If `bitsandbytes` complains, `Runtime → Restart session`, then
   re-run.
3. Run the **§1.3 `HF_TOKEN`** cell — optional, but do add the secret if you have one.
4. Get the script into the notebook — upload
   [`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](code/ex1_reproduce_KDA/run_qwen7b_eval.py)
   (§1.5), or run the `%%writefile` cell in [Appendix A](#appendix-a--entry-1s-script-inline)
   if you have no repository access.
5. **Run the evaluation.** Use `4bit` — it matches the local run, per the §1.4 policy, and on a
   free T4 it is both smaller *and* faster than `8bit`:

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --precision 4bit --batch-size 16 --out-dir /content/kda_out
   ```

   On a Pro+ A100 you can raise the batch size; keep the precision:

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --precision 4bit --batch-size 32 --out-dir /content/kda_out
   ```

   `--precision 8bit` / `bf16` also run, and `auto` will pick for you — but all three forfeit the
   diff against the local Qwen3-4B run (§1.4). Only use them if you are not making that comparison,
   and say so in the write-up:

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --out-dir /content/kda_out
   ```

6. **Smoke test first.** Before committing to a full run, confirm the whole path works on
   20 samples (~2 minutes including the model download):

   ```python
   !python run_qwen7b_eval.py --datasets sciq --limit 20 --precision 4bit --batch-size 4 --out-dir /content/kda_smoke
   ```

7. **Download the results** — see [§2.4](#24-outputs) and [§1.6](#16-getting-results-back-out).

> **Colab disconnects.** A free-tier session idles out after ~90 minutes and is capped
> around 12 hours. The full run fits comfortably, but keep the tab active. If you are
> worried, run the two datasets as separate invocations (`--datasets obqa`, then
> `--datasets sciq`) writing to different `--out-dir` values, or mount Drive first:
>
> ```python
> from google.colab import drive; drive.mount('/content/drive')
> # then pass --out-dir /content/drive/MyDrive/kda_out
> ```

---

### 2.2 Step-by-step: Kaggle

1. **New Notebook** → right panel → `Accelerator` → **GPU T4 x2** → `Internet` → **On**
   (required to download the model and datasets).
2. Run the **§1.2 install** cell, then `Run → Restart & Clear Cell Outputs` if `bitsandbytes`
   fails to import, and re-run.
3. Run the **§1.3** cell with the Kaggle secrets snippet.
4. Get the script into the notebook — upload
   [`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](code/ex1_reproduce_KDA/run_qwen7b_eval.py)
   (§1.5), or run the `%%writefile` cell in [Appendix A](#appendix-a--entry-1s-script-inline)
   if you have no repository access.
5. **Run the evaluation.** Keep `4bit` for comparability (§1.4):

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --precision 4bit --batch-size 16 --out-dir /kaggle/working/kda_out
   ```

   > With two T4s `device_map="auto"` shards the model across both, so `--precision fp16` fits
   > without quantisation. It is the faithful-to-full-precision option — and it forfeits the diff
   > against the local NF4 run. Comparability wins unless you have a reason it should not.

6. Anything written under **`/kaggle/working/`** is saved as notebook output. Use
   `Save Version → Save & Run All (Commit)` for an unattended run; results appear under the
   version's **Output** tab when it finishes.

> **Kaggle quotas.** GPU time is capped at ~30 hours/week and a single session at 12 hours.
> The full 1,384-sample run uses well under an hour.

---

### 2.3 Scoring the committed datasets

By default the script rebuilds SciQ and OpenBookQA from the Hub, applying the same
formatting as [`code/pre_data/`](code/pre_data): drop SciQ rows with an empty
`support`, join OpenBookQA's `fact1` from the `additional` config, and shuffle the four
options with a fixed seed.

To score **exactly** the files this repository already committed — guaranteeing the cloud
run and the local Qwen3-4B run see identical samples in identical option order — upload
them and pass `--local-data`.

**Colab:**

```python
from google.colab import files
uploaded = files.upload()   # select sciq_test_full.json and obqa_test_full.json
```

**Kaggle:** `Add Data → Upload → New Dataset`, then the files land under
`/kaggle/input/<your-dataset-slug>/`.

Then:

```python
!python run_qwen7b_eval.py \
    --datasets obqa sciq \
    --local-data sciq=./sciq_test_full.json obqa=./obqa_test_full.json \
    --precision 4bit --batch-size 16 --out-dir /content/kda_out
```

The two files live at
[`datasets/sciq/sciq_test_full.json`](datasets/sciq/sciq_test_full.json) (884 samples)
and [`datasets/openbookqa/obqa_test_full.json`](datasets/openbookqa/obqa_test_full.json)
(500 samples).

> **Use `--local-data` if you intend to compare against the Qwen3-4B numbers.** The rebuilt
> datasets use the same seed and logic, but pinning the actual files removes any doubt
> about dataset drift on the Hub.

---

### 2.4 Outputs

The run writes four files into `--out-dir`:

| File | Contents |
|---|---|
| `kda_qwen2.5_7b_results.json` | Full per-sample records + the summary block (config, memory, per-dataset metrics) |
| `kda_qwen2.5_7b_records.csv` | One row per sample: probabilities, predictions, `r_q`, `r_q_plus_f`, bucket |
| `kda_qwen2.5_7b_summary.csv` | One row per dataset: KDA_disc, KDA_cont, accuracies, bucket counts |
| `kda_qwen2.5_7b_eval.log` | Full execution log (DEBUG level in the file, INFO on stdout) |

Inspect the summary inline:

```python
import pandas as pd
pd.read_csv("/content/kda_out/kda_qwen2.5_7b_summary.csv")
```

**Download from Colab:**

```python
import shutil
from google.colab import files
shutil.make_archive("/content/kda_out", "zip", "/content/kda_out")
files.download("/content/kda_out.zip")
```

**Download from Kaggle:** files under `/kaggle/working/` appear in the notebook's **Output**
panel — click the download icon, or `Save Version` and grab the artefacts from the version
page.

To fold the numbers back into this repository, drop the JSON into
`results/ex1_reproduce_KDA_w_modernLLM/` — the stage directory that holds every modern-LLM
evaluation — and record the comparison in `docs/`:

```bash
cp ~/Downloads/kda_qwen2.5_7b_results.json results/ex1_reproduce_KDA_w_modernLLM/
```

---

---

## 3. Entry 2 — Persona simulation (SciQ + OBQA)

**What it establishes.** Whether the emission-order artifact is model-specific. On Qwen3-4B the
"ability gap" produced by joint multi-persona prompting turned out to be positional: reversing
the tier order moved the SciQ gap from +0.342 to +0.639 and flipped OBQA from −0.228 to +0.408,
because the **last-emitted tier** is the one pushed off the consensus. If Qwen2.5-7B shows the
same order dependence, it is a property of the prompting scheme; if not, it is a Qwen3-4B quirk.

**What to run.** Upload the repo (§1.5), then:

```bash
python code/ex3_student_simulation/run_student_simulation.py \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --out-dir results/ex3_student_simulation \
    --tag qwen25 \
    --log-file results/ex3_student_simulation/student_simulation_qwen25.log
```

```bash
python code/ex3_student_simulation/run_student_simulation.py \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --paradigms joint --joint-order descending \
    --tag qwen25_order_desc \
    --log-file results/ex3_student_simulation/order_desc_qwen25.log
```

> **Note:** `--model-path` currently resolves through `utils.paths.resolve`, which treats a
> non-absolute value as repo-relative. A bare HuggingFace id such as `Qwen/Qwen2.5-7B-Instruct`
> will therefore be looked for on disk. Either download the model to `models/Qwen2.5-7B-Instruct`
> first (recommended — matches the local layout), or pass an absolute path to a snapshot
> directory.

**Runtime / VRAM:** Qwen3-4B took 40.4 min for 1,384 items × 2 paradigms at 1.79 s/item on a
4 GB laptop GPU. On a T4 with a 7B model at 4-bit expect **~70–100 min** for the main run plus
**~35–50 min** for the order control. Peak VRAM ~6 GB (joint prompts are long, ~400 tokens).

**Expected outputs:**

```
results/ex3_student_simulation/results_persona_simulation_sciq_qwen25.json
results/ex3_student_simulation/results_persona_simulation_obqa_qwen25.json
results/ex3_student_simulation/results_persona_simulation_sciq_qwen25_order_desc.json
results/ex3_student_simulation/results_persona_simulation_obqa_qwen25_order_desc.json
docs/student_persona_simulation_qwen2.5_7b_report.md
```

**Comparison analysis once both exist:**

1. Regenerate every table with
   `python code/ex3_student_simulation/summarize_simulation.py --tag qwen25`.
2. **Order-dependence diff** — the decisive comparison. Tabulate, for each model and dataset, the
   ascending and descending ability gaps. Qwen3-4B: SciQ +0.342 → +0.639, OBQA −0.228 → +0.408.
3. **Differentiation-pressure diff** — `joint_differentiation.last_tier_deviates_from_agreeing_earlier_tiers`
   (Qwen3-4B: 23.7% SciQ / 72.1% OBQA ascending; 71.4% / 66.7% descending) and
   `deviations_away_from_a_correct_consensus`.
4. **Position-collapse diff** — the answer-letter histogram χ² against the gold spread.
   Qwen3-4B joint-beginner put 54% of answers (63% of errors) on "A" for SciQ ascending, 53% on
   "D" for OBQA descending. Does the larger model collapse onto a letter too?

---

## 4. Entry 3 — Free-response **second judge** ⚠ CRITICAL PATH

**What it establishes.** The free-response experiment grades open-ended answers with an LLM judge.
The pilot found the judge decides **76–100%** of items and **flips 24.3% of its verdicts when
reference and student are swapped** — and bidirectional grading exposes rather than removes that
instability (band widths: SciQ A′ 0.120, OBQA A′ 0.240, OBQA B′ 0.360). Because the judge is
Qwen3-4B grading Qwen3-4B's own output, self-judging bias is a first-order threat. An independent
judge is needed before any headline number is published.

**Self-contained — paste directly into a notebook.** Needs only the pilot result files
(`results/ex4_free_response/results_free_response_{sciq,obqa}_pilot_bidir.json`) uploaded.

```python
import json, torch

JUDGE_SYSTEM = (
    "You are grading a short-answer science exam. Decide whether the student's answer means the "
    "same thing as the reference answer, in the context of the question. Ignore spelling, "
    "capitalisation, and phrasing. A more specific or more general answer counts as correct only "
    "if it identifies the same thing. Reply with exactly one word: CORRECT or INCORRECT."
)

def judge_user_turn(question, reference, student):
    return (f"Question: {str(question).strip()}\n"
            f"Reference answer: {str(reference).strip()}\n"
            f"Student answer: {str(student).strip()}")

# First tokens of the two verdicts; they must differ for the two-way logit gate to work.
VERDICT_IDS = {w: tokenizer.encode(w, add_special_tokens=False)[0] for w in ("CORRECT", "INCORRECT")}
assert VERDICT_IDS["CORRECT"] != VERDICT_IDS["INCORRECT"], "verdicts share a first token"

@torch.no_grad()
def judge(question, reference, student):
    """Restrict the final-position logits to the two verdict tokens and softmax over them.

    Same protocol as the local Qwen3-4B judge: deterministic, calibrated, and with no free-text
    parsing of the judge's own output."""
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": judge_user_turn(question, reference, student)}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[0, -1, :].float()
    pair = torch.stack([logits[VERDICT_IDS["CORRECT"]], logits[VERDICT_IDS["INCORRECT"]]])
    p = torch.softmax(pair, dim=0).tolist()
    return {"verdict": "CORRECT" if p[0] > p[1] else "INCORRECT", "p_correct": p[0]}

out = {}
for ds in ("sciq", "obqa"):
    payload = json.load(open(f"results_free_response_{ds}_pilot_bidir.json"))
    rows = []
    for cell in ("A_prime", "B_prime"):
        for r in payload["results"][cell]:
            if r["match_stage"] != "judge":
                continue          # deterministic matches need no judging
            fwd = judge(r["question"], r["gold_answer"], r["prediction"])
            rev = judge(r["question"], r["prediction"], r["gold_answer"])   # swapped
            rows.append({
                "id": r["id"], "cell": cell,
                "question": r["question"],
                "gold_answer": r["gold_answer"], "prediction": r["prediction"],
                "qwen25_forward": fwd, "qwen25_reverse": rev,
                "qwen25_agree": fwd["verdict"] == rev["verdict"],
                "qwen3_forward": r["judge"]["forward"]["verdict"],
                "qwen3_reverse": r["judge"]["reverse"]["verdict"],
                "qwen3_agree": r["judge_agree"],
            })
    out[ds] = rows
    print(ds, len(rows), "items judged")

json.dump({"judge_model": MODEL_ID, "results": out},
          open("judge_qwen2.5_7b_free_response_pilot.json", "w"), indent=2)
```

**Runtime / VRAM:** 74 judged items × 2 directions = 148 forward passes — **under 2 minutes**.
~5 GB VRAM. This is by far the cheapest entry and it unblocks the most.

**Expected output:**

```
results/ex4_free_response/judge_qwen2.5_7b_free_response_pilot.json
```

**Comparison analysis once both exist — the cross-tabulation to compute:**

1. **Order-instability, per judge.** Flip rate under swap for each model. Qwen3-4B: **24.3%**
   overall (SciQ A′ 15.8%, SciQ B′ 0.0%, OBQA A′ 24.0%, OBQA B′ 39.1%). If Qwen2.5-7B flips at a
   similar rate, the instability is a property of the *task framing* (probably the "more specific
   or more general" clause, which is inherently directional) and the prompt must be rewritten. If
   it flips much less, it should simply replace Qwen3-4B as the judge.
2. **Inter-judge agreement, forward verdicts.** A 2×2 table over the 74 items:

   | | Qwen2.5 CORRECT | Qwen2.5 INCORRECT |
   |---|---|---|
   | **Qwen3 CORRECT** | | |
   | **Qwen3 INCORRECT** | | |

   Report raw agreement **and Cohen's κ**. Reuse `code/ex4_free_response/compute_kappa.py` by
   writing the two judges' verdicts into the `human_verdict` / `judge_forward` columns of a
   validation sheet — the same κ machinery and the same 0.70 threshold apply.
3. **Self-judging bias.** Qwen3-4B graded its *own* generations. Split the agreement table by
   whether Qwen3-4B said CORRECT: if Qwen2.5-7B systematically disagrees where Qwen3-4B accepted
   its own answer, that is the bias, and its size is the correction to apply.
4. **Effect on the reported band.** Recompute `acc_band_low` / `acc_band_high` per cell using
   Qwen2.5-7B as judge and put the two bands side by side. If they overlap, the free-response
   headline is robust to judge choice; if not, the experiment needs a human-adjudicated core.

---

## 5. Entry 4 — Free-response generation (optional, not blocking)

**What it establishes.** Whether the recognition-vs-recall gap is model-specific. Qwen3-4B pilot
(n=25): SciQ A′ band [0.680, 0.800] against MCQ 0.954; OBQA A′ band [0.080, 0.320] against MCQ
0.826 — but OBQA must be read against its own B′ ceiling, not against MCQ.

**What to run.** Repo upload (§1.5), then:

```bash
python code/ex4_free_response/run_free_response.py \
    --model-path <path to Qwen2.5-7B snapshot> --tag qwen25_pilot --limit 25 \
    --log-file results/ex4_free_response/free_response_qwen25_pilot.log
```

**Runtime:** Qwen3-4B managed 0.71–0.77 s/item locally; expect ~1.5–2 s/item for 7B on a T4, so
~5 min for a 25-item pilot and ~90 min for the full 1,384 × 2 cells.

**Expected outputs:** `results/ex4_free_response/results_free_response_{sciq,obqa}_qwen25_pilot.json`

**Comparison analysis:** diff `acc_band_low`/`acc_band_high` per cell and, critically, diff each
dataset's **B′ ceiling** — if Qwen2.5-7B also tops out near 0.44 on OBQA B′, the ceiling is a
property of OpenBookQA's annotator phrasing rather than of the model.

---

## 6. Troubleshooting

Applies to every entry — all four load the same model on the same platforms.

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` during scoring | Batch too large for the GPU | The script auto-halves the batch and retries. To avoid it entirely, lower `--batch-size` or drop to `--precision 4bit`. |
| `CUDA out of memory` while *loading* | Precision too high for the card | Use `8bit` or `4bit`. A single T4 cannot hold 7B in fp16 with room to run. |
| `bitsandbytes` → `CUDA Setup failed` | Wheel imported before the CUDA runtime settled | `Runtime → Restart session`, then re-run the install cell. |
| `ValueError: Cannot use bf16 ... device does not support` | T4 has no bfloat16 | Use `--precision fp16` / `8bit` / `4bit`, or leave `auto`. |
| Accuracy near 25% (chance) on all four options | Chat template not applied, or right padding | Confirm `tokenizer.padding_side = "left"` and that `apply_chat_template(..., add_generation_prompt=True)` ran. Both are handled by the script. |
| `Letter 'A' has no single-token spelling` | Non-Qwen tokenizer that splits bare letters | Score per-option sequence likelihood instead, or pick a tokenizer with single-token letters. |
| Model download is very slow | Unauthenticated rate limit | Set `HF_TOKEN` (§1.3). |
| `datasets` fails on `allenai/openbookqa` | Missing config name | The `additional` config carries `fact1`; the script requests it explicitly. |
| Kaggle: `ConnectionError` on download | Internet toggle off | Right panel → `Internet → On`. |

### Sanity checks worth running

```python
# 1. The four option letters must be single tokens.
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
for letter in "ABCD":
    print(letter, tok.encode(letter, add_special_tokens=False),
                  tok.encode(" " + letter, add_special_tokens=False))

# 2. Eyeball one rendered prompt end-to-end.
import json
sample = {"passage": "Dissociation is the separation of ions that occurs when a solid ionic compound dissolves.",
          "question": "What is the separation of ions that occurs when a solid ionic compound dissolves?",
          "options": ["dissociation", "combustion", "inflammation", "decomposition"], "answer_idx": 0}
import run_qwen7b_eval as R
print(R.render_chat(tok, R.build_prompt(sample, with_fact=True)))
```

A correct render ends with the assistant turn opened and no content after it — that final
position is where the option-letter logits are read.

---

---

## 7. Conventions for returned files

- **Naming mirrors the Qwen3-4B artefact exactly**, substituting `qwen2.5_7b` for `qwen3_4b`, so
  the two are trivially diffable (`kda_qwen2.5_7b_results.json` ↔ `kda_qwen3_4b_results.json`).
- Keep the `summary` block the scripts emit — the comparison analyses read it directly.
- Include the run log; the timing and VRAM lines belong in the comparison report.
- Record the **exact model revision** (`model.config._name_or_path` and the HF commit hash if
  pinned) in the summary, so a later re-run is identifiable.

---

## 8. Changelog

| Round | Added |
|---|---|
| 2026-09-08 | Gate (a), the human validation gate, cleared: κ = 0.754, `gate_passed = true`. Entry 3 still not run; the full free-response run stays blocked on it. |
| 2026-09-04 | File created. Entries 1–4 populated from an exhaustive audit of every LLM-invoking script (§0.1). Entry 3 flagged as the critical path blocking the full free-response run. |

**When appending:** add a row to §0's status board, a full entry section, the comparison analysis,
and a changelog row. New Qwen3-4B work is not complete until its Qwen2.5 counterpart is queued
here.
---

## Appendix A — Entry 1's script, inline

> **Canonical source:** [`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](code/ex1_reproduce_KDA/run_qwen7b_eval.py).
> That file is the version to trust and the one to upload if you have the repository. This
> appendix exists so Entry 1 can be run with **no repository access at all** — paste the cell
> below into a notebook and it writes the script for you.
>
> The two are not byte-identical: as of 2026-09-11 the `.py` file carries a fuller module
> docstring, a `chunked()` helper and three logging constants that this inline copy omits. All
> 17 functions the two share have identical signatures, so the inline copy is a lean but
> faithful subset — not a stale fork. **If you change one, change both**, or delete this
> appendix and point at the `.py` instead.

Run the cell below to write the evaluation script to disk. This is the complete, runnable
program: dataset construction, model loading, batched logit scoring, KDA computation, and
JSON/CSV export.

```python
%%writefile run_qwen7b_eval.py
"""Standalone KDA evaluation for Qwen2.5-7B-Instruct on Colab / Kaggle."""
from __future__ import annotations
import argparse, csv, json, logging, os, platform, random, statistics, sys, time
from typing import Dict, List, Optional, Sequence, Tuple
import torch

LETTERS = ("A", "B", "C", "D")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = ("You are a careful student taking a multiple-choice exam. "
                 "Answer with a single letter and nothing else.")
DATASET_NAMES = {"sciq": "allenai/sciq", "obqa": "allenai/openbookqa"}
FACT_FIELDS = {"sciq": "support paragraph", "obqa": "fact1"}


def setup_logging(log_path, verbose=False):
    root = logging.getLogger(); root.setLevel(logging.DEBUG)
    for h in list(root.handlers): root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.DEBUG if verbose else logging.INFO); sh.setFormatter(fmt)
    root.addHandler(fh); root.addHandler(sh)
    for n in ("httpx","httpcore","urllib3","filelock","huggingface_hub","transformers","datasets","fsspec","accelerate","bitsandbytes"):
        logging.getLogger(n).setLevel(logging.WARNING)
    return logging.getLogger("kda_qwen7b")


# ---------------------------------------------------------------- datasets
def build_sciq(split, seed, logger):
    """allenai/sciq -> KDA format. The `support` paragraph is the target fact."""
    from datasets import load_dataset
    raw = load_dataset("allenai/sciq", split=split); rng = random.Random(seed)
    samples, dropped = [], 0
    for row_index, row in enumerate(raw):
        support = (row.get("support") or "").strip()
        if not support: dropped += 1; continue
        correct = (row["correct_answer"] or "").strip()
        options = [correct, (row["distractor1"] or "").strip(),
                   (row["distractor2"] or "").strip(), (row["distractor3"] or "").strip()]
        rng.shuffle(options)
        samples.append({"passage": support, "question": (row["question"] or "").strip(),
                        "options": options, "answer_idx": options.index(correct),
                        "correct_answer": correct, "id": len(samples), "dataset_row": row_index})
    logger.info("  SciQ %s: %d samples (%d dropped for empty support)", split, len(samples), dropped)
    return samples


def build_obqa(split, seed, logger):
    """allenai/openbookqa -> KDA format. `fact1` (the `additional` config) is the fact."""
    from datasets import load_dataset
    raw = load_dataset("allenai/openbookqa", "additional", split=split); rng = random.Random(seed)
    samples, dropped = [], 0
    for row_index, row in enumerate(raw):
        fact = (row.get("fact1") or "").strip(); choices = row["choices"]
        texts = [t.strip() for t in choices["text"]]; labels = list(choices["label"]); key = row["answerKey"]
        if not fact or key not in labels or len(texts) != 4: dropped += 1; continue
        correct = texts[labels.index(key)]; options = list(texts); rng.shuffle(options)
        samples.append({"passage": fact, "question": (row["question_stem"] or "").strip(),
                        "options": options, "answer_idx": options.index(correct),
                        "correct_answer": correct, "id": len(samples), "dataset_row": row_index})
    logger.info("  OpenBookQA %s: %d samples (%d dropped)", split, len(samples), dropped)
    return samples


def load_local_json(path, logger):
    with open(path, encoding="utf-8") as h: samples = json.load(h)
    logger.info("  loaded %d samples from %s", len(samples), path); return samples


DATASET_BUILDERS = {"sciq": build_sciq, "obqa": build_obqa}


# ---------------------------------------------------------------- prompting
def build_prompt(sample, with_fact):
    lines = []
    if with_fact:
        lines += [f"Fact: {(sample.get('passage') or '').strip()}", ""]
    lines.append(f"Question: {sample['question'].strip()}")
    for letter, option in zip(LETTERS, sample["options"]):
        lines.append(f"{letter}. {str(option).strip()}")
    lines += ["", "Answer with a single letter (A, B, C, or D)."]
    return "\n".join(lines)


def render_chat(tokenizer, user_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def letter_token_ids(tokenizer, logger):
    """All single-token spellings of each option letter (bare and space-prefixed)."""
    variants = {}
    for letter in LETTERS:
        ids = []
        for spelling in (letter, f" {letter}"):
            enc = tokenizer.encode(spelling, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in ids: ids.append(enc[0])
        if not ids: raise RuntimeError(f"Letter {letter!r} has no single-token spelling.")
        variants[letter] = ids
    logger.info("  option-letter token ids: %s", variants); return variants


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def score_batch(model, tokenizer, prompts, letter_ids):
    """Batched forward pass -> per-prompt probability over the four option letters.

    Uses LEFT padding so index -1 is the real final token of every row in the batch.
    """
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True,
                    max_length=4096, add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[:, -1, :].float()
    per_letter = torch.stack(
        [torch.logsumexp(logits[:, torch.tensor(letter_ids[l], device=logits.device)], dim=-1)
         for l in LETTERS], dim=-1)
    return torch.softmax(per_letter, dim=-1).tolist()


# ---------------------------------------------------------------- metrics
def compute_metrics(records):
    n = len(records)
    if n == 0: return {"n_samples": 0}
    r_q = [r["r_q"] for r in records]; r_qf = [r["r_q_plus_f"] for r in records]
    p_q = [r["p_correct_without_fact"] for r in records]; p_qf = [r["p_correct_with_fact"] for r in records]
    disc_den = sum(1 - r for r in r_q); disc_num = sum((1 - a) * b for a, b in zip(r_q, r_qf))
    cont_den = sum(1.0 - p for p in p_q); cont_num = sum((1.0 - a) * b for a, b in zip(p_q, p_qf))
    buckets = {"both_correct": 0, "wrong_to_correct": 0, "both_wrong": 0, "correct_to_wrong": 0}
    for a, b in zip(r_q, r_qf):
        if a == 1 and b == 1: buckets["both_correct"] += 1
        elif a == 0 and b == 1: buckets["wrong_to_correct"] += 1
        elif a == 0 and b == 0: buckets["both_wrong"] += 1
        else: buckets["correct_to_wrong"] += 1
    return {"n_samples": n,
            "kda_disc": (disc_num / disc_den) if disc_den else None,
            "kda_disc_numerator": disc_num, "kda_disc_denominator": disc_den,
            "kda_cont": (cont_num / cont_den) if cont_den > 0 else None,
            "kda_cont_numerator": cont_num, "kda_cont_denominator": cont_den,
            "accuracy_without_fact": sum(r_q) / n, "accuracy_with_fact": sum(r_qf) / n,
            "accuracy_gain": (sum(r_qf) - sum(r_q)) / n, "memorization_rate": sum(r_q) / n,
            "mean_p_correct_without_fact": statistics.fmean(p_q),
            "mean_p_correct_with_fact": statistics.fmean(p_qf),
            "buckets": {k: {"count": v, "percentage": round(100.0 * v / n, 4)} for k, v in buckets.items()}}


# ---------------------------------------------------------------- model
def resolve_precision(requested, logger):
    """Pick a precision that actually fits the attached accelerator."""
    if requested != "auto": return requested
    if not torch.cuda.is_available():
        logger.warning("No CUDA device -- falling back to fp32 on CPU (very slow)."); return "fp32"
    total_gb = sum(torch.cuda.get_device_properties(i).total_memory
                   for i in range(torch.cuda.device_count())) / 1024**3
    bf16_ok = torch.cuda.is_bf16_supported()
    logger.info("  detected %.1f GB across %d GPU(s); bf16 supported: %s",
                total_gb, torch.cuda.device_count(), bf16_ok)
    if total_gb >= 30 and bf16_ok: return "bf16"   # A100 / H100
    if total_gb >= 28: return "fp16"               # 2x T4 (Kaggle), sharded
    if total_gb >= 15: return "8bit"               # single T4 16GB (Colab free)
    return "4bit"


def load_model(model_id, precision, token, logger):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    logger.info("Loading %s at precision '%s'", model_id, precision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.padding_side = "left"                       # REQUIRED: we read logits at -1
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": "auto", "token": token}
    if precision == "bf16": kwargs["dtype"] = torch.bfloat16
    elif precision == "fp16": kwargs["dtype"] = torch.float16
    elif precision == "fp32": kwargs["dtype"] = torch.float32; kwargs["device_map"] = None
    elif precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True); kwargs["dtype"] = torch.float16
    elif precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        kwargs["dtype"] = torch.float16
    else: raise ValueError(f"unknown precision {precision!r}")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs); model.eval()
    load_seconds = time.time() - started
    memory = {"precision": precision, "load_seconds": round(load_seconds, 2)}
    if torch.cuda.is_available():
        memory["allocated_after_load_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        memory["reserved_after_load_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
        logger.info("  loaded in %.1fs | %.2f GB allocated", load_seconds, memory["allocated_after_load_gb"])
    return model, tokenizer, memory


def _fmt(v): return "n/a (zero denominator)" if v is None else f"{v:.4f}"


# ---------------------------------------------------------------- driver
def evaluate_dataset(model, tokenizer, letter_ids, key, samples, batch_size, logger):
    logger.info(""); logger.info("=" * 96)
    logger.info("DATASET %s -- %d samples, batch size %d", key.upper(), len(samples), batch_size)
    logger.info("=" * 96)
    prompts_wof = [render_chat(tokenizer, build_prompt(s, False)) for s in samples]
    prompts_wf = [render_chat(tokenizer, build_prompt(s, True)) for s in samples]

    def run_phase(prompts, label):
        out, started, current, index = [], time.time(), batch_size, 0
        while index < len(prompts):
            batch = prompts[index:index + current]
            try:
                out.extend(score_batch(model, tokenizer, batch, letter_ids))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current == 1: raise
                current = max(1, current // 2)
                logger.warning("  OOM -- reducing batch size to %d and retrying", current); continue
            index += len(batch)
            if len(out) % max(batch_size * 10, 100) < current or index >= len(prompts):
                el = time.time() - started
                logger.info("  %s %5d/%5d (%.1f%%) | %.3fs/sample | ETA %.1f min", label, len(out),
                            len(prompts), 100.0 * len(out) / len(prompts), el / max(len(out), 1),
                            (el / max(len(out), 1)) * (len(prompts) - len(out)) / 60.0)
        return out, time.time() - started

    probs_wof, sec_wof = run_phase(prompts_wof, "phase1 (no fact)  ")
    probs_wf, sec_wf = run_phase(prompts_wf, "phase2 (with fact)")

    records = []
    for sample, p_wof, p_wf in zip(samples, probs_wof, probs_wf):
        ai = int(sample["answer_idx"])
        pred_wof = int(max(range(4), key=lambda i: p_wof[i]))
        pred_wf = int(max(range(4), key=lambda i: p_wf[i]))
        records.append({"id": sample.get("id"), "dataset_row": sample.get("dataset_row"),
                        "question": sample["question"], "options": sample["options"], "answer_idx": ai,
                        "correct_answer": sample.get("correct_answer", sample["options"][ai]),
                        "passage": sample.get("passage", ""),
                        "probs_without_fact": p_wof, "probs_with_fact": p_wf,
                        "predicted_idx_without_fact": pred_wof, "predicted_idx_with_fact": pred_wf,
                        "p_correct_without_fact": p_wof[ai], "p_correct_with_fact": p_wf[ai],
                        "r_q": int(pred_wof == ai), "r_q_plus_f": int(pred_wf == ai)})

    metrics = compute_metrics(records); total = sec_wof + sec_wf
    metrics.update({"dataset_key": key, "dataset_name": DATASET_NAMES.get(key, key),
                    "fact_field": FACT_FIELDS.get(key, "passage"),
                    "scoring_seconds": round(total, 2),
                    "seconds_per_sample": round(total / max(len(samples), 1), 4)})
    logger.info("")
    logger.info("  KDA_disc         : %s", _fmt(metrics["kda_disc"]))
    logger.info("  KDA_cont         : %s", _fmt(metrics["kda_cont"]))
    logger.info("  Acc without fact : %.4f", metrics["accuracy_without_fact"])
    logger.info("  Acc with fact    : %.4f (delta %+.4f)", metrics["accuracy_with_fact"], metrics["accuracy_gain"])
    logger.info("  Memorized (r^q=1): %.2f%%", 100.0 * metrics["memorization_rate"])
    logger.info("  buckets          : %s", {k: v["count"] for k, v in metrics["buckets"].items()})
    return {"metrics": metrics, "records": records}


def export_csv(out_dir, per_dataset, all_records, logger):
    """Flat per-sample CSV plus a one-row-per-dataset summary CSV."""
    rec_path = os.path.join(out_dir, "kda_qwen2.5_7b_records.csv")
    with open(rec_path, "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["dataset","id","dataset_row","question","correct_answer","answer_idx",
                    "option_A","option_B","option_C","option_D",
                    "p_correct_without_fact","p_correct_with_fact",
                    "predicted_idx_without_fact","predicted_idx_with_fact",
                    "r_q","r_q_plus_f","bucket","passage"])
        for key, records in all_records.items():
            for r in records:
                if r["r_q"] == 1 and r["r_q_plus_f"] == 1: b = "both_correct"
                elif r["r_q"] == 0 and r["r_q_plus_f"] == 1: b = "wrong_to_correct"
                elif r["r_q"] == 0 and r["r_q_plus_f"] == 0: b = "both_wrong"
                else: b = "correct_to_wrong"
                o = list(r["options"]) + [""] * (4 - len(r["options"]))
                w.writerow([key, r["id"], r["dataset_row"], r["question"], r["correct_answer"],
                            r["answer_idx"], o[0], o[1], o[2], o[3],
                            f'{r["p_correct_without_fact"]:.6f}', f'{r["p_correct_with_fact"]:.6f}',
                            r["predicted_idx_without_fact"], r["predicted_idx_with_fact"],
                            r["r_q"], r["r_q_plus_f"], b, r["passage"]])
    sum_path = os.path.join(out_dir, "kda_qwen2.5_7b_summary.csv")
    with open(sum_path, "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["dataset","n_samples","kda_disc","kda_cont","accuracy_without_fact",
                    "accuracy_with_fact","accuracy_gain","memorization_rate",
                    "mean_p_correct_without_fact","mean_p_correct_with_fact",
                    "both_correct","wrong_to_correct","both_wrong","correct_to_wrong","seconds_per_sample"])
        for key, m in per_dataset.items():
            w.writerow([key, m["n_samples"],
                        "" if m["kda_disc"] is None else f'{m["kda_disc"]:.6f}',
                        "" if m["kda_cont"] is None else f'{m["kda_cont"]:.6f}',
                        f'{m["accuracy_without_fact"]:.6f}', f'{m["accuracy_with_fact"]:.6f}',
                        f'{m["accuracy_gain"]:.6f}', f'{m["memorization_rate"]:.6f}',
                        f'{m["mean_p_correct_without_fact"]:.6f}', f'{m["mean_p_correct_with_fact"]:.6f}',
                        m["buckets"]["both_correct"]["count"], m["buckets"]["wrong_to_correct"]["count"],
                        m["buckets"]["both_wrong"]["count"], m["buckets"]["correct_to_wrong"]["count"],
                        m.get("seconds_per_sample", "")])
    logger.info("CSV records : %s", rec_path); logger.info("CSV summary : %s", sum_path)
    return rec_path, sum_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Standalone KDA evaluation for Qwen2.5-7B-Instruct.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--datasets", nargs="+", default=["obqa", "sciq"], choices=["obqa", "sciq"])
    p.add_argument("--split", default="test")
    p.add_argument("--local-data", nargs="+", default=None, metavar="KEY=PATH")
    p.add_argument("--precision", default="auto", choices=["auto","bf16","fp16","8bit","4bit","fp32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", default="./kda_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out_dir); os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(os.path.join(out_dir, "kda_qwen2.5_7b_eval.log"), args.verbose)
    started = time.time(); token = args.token or os.environ.get("HF_TOKEN") or None

    logger.info("=" * 96); logger.info("KDA EVALUATION -- %s", args.model_id); logger.info("=" * 96)
    logger.info("Python   : %s", platform.python_version())
    logger.info("torch    : %s (cuda %s)", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            logger.info("GPU %d    : %s (%.1f GB)", i, pr.name, pr.total_memory / 1024**3)
    logger.info("HF token : %s", "provided" if token else "none (public model)")
    logger.info("Output   : %s", out_dir); logger.info("")

    local_map = {}
    for item in args.local_data or []:
        if "=" not in item: logger.error("--local-data expects KEY=PATH, got %r", item); return 1
        k, v = item.split("=", 1); local_map[k.strip()] = v.strip()

    logger.info("Preparing datasets ...")
    datasets_map = {}
    for key in args.datasets:
        s = load_local_json(local_map[key], logger) if key in local_map \
            else DATASET_BUILDERS[key](args.split, args.seed, logger)
        datasets_map[key] = s[: args.limit] if args.limit else s

    precision = resolve_precision(args.precision, logger)
    model, tokenizer, memory = load_model(args.model_id, precision, token, logger)
    letter_ids = letter_token_ids(tokenizer, logger)

    per_dataset, all_records = {}, {}
    for key, samples in datasets_map.items():
        out = evaluate_dataset(model, tokenizer, letter_ids, key, samples, args.batch_size, logger)
        per_dataset[key] = out["metrics"]; all_records[key] = out["records"]

    if torch.cuda.is_available():
        memory["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        memory["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

    payload = {"summary": {"experiment": "kda_evaluation_qwen2.5_7b_instruct",
                           "model": args.model_id, "precision": precision,
                           "batch_size": args.batch_size, "split": args.split, "seed": args.seed,
                           "limit": args.limit or None,
                           "formulas": {"kda_disc": "sum_i (1 - r_i^q) * r_i^{q+f} / sum_i (1 - r_i^q)",
                                        "kda_cont": "sum_i (1 - P_i(R^q=1)) * P_i(R^{q+f}=1) / sum_i (1 - P_i(R^q=1))"},
                           "memory": memory, "per_dataset": per_dataset,
                           "total_runtime_seconds": round(time.time() - started, 2)},
               "results": all_records}
    json_path = os.path.join(out_dir, "kda_qwen2.5_7b_results.json")
    with open(json_path, "w", encoding="utf-8") as h: json.dump(payload, h, ensure_ascii=False, indent=2)
    export_csv(out_dir, per_dataset, all_records, logger)

    logger.info(""); logger.info("=" * 96); logger.info("SUMMARY"); logger.info("=" * 96)
    logger.info("  %-8s %10s %10s %10s %10s %12s", "dataset","KDA_disc","KDA_cont","Acc_wof","Acc_wf","memorized")
    for key, m in per_dataset.items():
        logger.info("  %-8s %10s %10s %10.4f %10.4f %11.2f%%", key, _fmt(m["kda_disc"]), _fmt(m["kda_cont"]),
                    m["accuracy_without_fact"], m["accuracy_with_fact"], 100.0 * m["memorization_rate"])
    logger.info(""); logger.info("JSON     : %s", json_path)
    logger.info("Runtime  : %.1f min", payload["summary"]["total_runtime_seconds"] / 60.0)
    logger.info("Done."); return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---
