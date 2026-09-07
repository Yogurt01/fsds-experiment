# `RUN_QWEN2.5.md` — Cloud runbook for Qwen2.5-7B counterparts

**Living checklist.** Every round that completes new Qwen3-4B-dependent work appends an entry
here. Do not treat it as finished.

**Why this file exists.** The local machine is an RTX 3050 Laptop (4 GB). `Qwen3-4B-Instruct-2507`
fits under 4-bit NF4 (2.66 GB weights, 3.41 GB peak). `Qwen2.5-7B-Instruct` (7.62B params) does
not fit with useful headroom, so every Qwen2.5 counterpart runs on Colab/Kaggle instead.

**Portability.** Each entry is written to be pasted into a fresh notebook with no other project
context. Where a repository script is needed, §1.3 shows how to get it there.

---

## 0. Status board

| # | Task | Qwen3-4B status | Qwen2.5-7B status | Blocking? |
|---|---|---|---|---|
| 1 | KDA saturation evaluation | ✅ done — `docs/kda_qwen3_4b_evaluation_report.md` | ⬜ **not run** | no |
| 2 | Persona simulation (SciQ + OBQA) | ✅ done — `docs/student_persona_simulation_report.md` | ⬜ **not run** | no |
| 3 | Free-response **second judge** | ✅ Qwen3-4B is judge — pilot done | ⬜ **not run** | **YES — blocks the full free-response run** |
| 4 | Free-response **generation** (cells A′/B′) | ✅ pilot done (n=25) | ⬜ not run (optional) | no |

**Only entry 3 is on the critical path.** The full free-response run is blocked on
(a) the human validation gate (κ ≥ 0.70, see `code/ex4_free_response/compute_kappa.py`) and
(b) entry 3 below.

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

## 1. Common setup (run once per notebook)

### 1.1 Install

```python
!pip install -q -U "transformers>=4.44" accelerate bitsandbytes datasets
import torch, transformers
print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))
```

### 1.2 Load the model (4-bit NF4 — matches the local Qwen3-4B configuration)

Use 4-bit so quantisation is not a confound when diffing against the local runs. Both models are
then NF4 + fp16 compute.

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

| Precision | Weights | T4 16GB | A100 40GB |
|---|---:|---|---|
| `bf16`/`fp16` | ~15.2 GB | ❌ | ✅ |
| `8bit` | ~8.0 GB | ✅ | ✅ |
| **`4bit` NF4 (use this)** | **~4.5 GB** | ✅ ample | ✅ |

### 1.3 Getting project files into the notebook

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

## 2. Entry 1 — KDA saturation evaluation

**What it establishes.** Whether parametric saturation is specific to Qwen3-4B or general to
modern instruct models. Qwen3-4B reaches `Acc_wof` = **0.954** on SciQ / **0.826** on OBQA, which
is the motivating fact for the whole programme. If Qwen2.5-7B also saturates, the finding
generalises; if it does not, the local conclusion is model-specific.

**What to run.** The standalone script already exists and imports nothing from the repo:
[`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](code/ex1_reproduce_KDA/run_qwen7b_eval.py). Upload
that one file.

```bash
pip install -q -U transformers accelerate bitsandbytes datasets pandas
python run_qwen7b_eval.py --datasets sciq obqa --out-dir ./kda_out
```

The detailed step-by-step (Colab and Kaggle variants, troubleshooting) is in
[`docs/RUN_QWEN2.5_7B_CLOUD_GUIDE.md`](docs/RUN_QWEN2.5_7B_CLOUD_GUIDE.md), which remains valid;
this entry is the index card for it.

**Runtime / VRAM:** ~25–40 min for 1,384 items at 4-bit on a T4; ~5 GB VRAM.

**Expected outputs — download and place at:**

```
results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_results.json      (parallel to kda_qwen3_4b_results.json)
results/ex1_reproduce_KDA_w_modernLLM/kda_qwen2.5_7b_eval.log
docs/kda_qwen2.5_7b_evaluation_report.md                            (parallel to kda_qwen3_4b_evaluation_report.md)
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

## 3. Entry 2 — Persona simulation (SciQ + OBQA)

**What it establishes.** Whether the emission-order artifact is model-specific. On Qwen3-4B the
"ability gap" produced by joint multi-persona prompting turned out to be positional: reversing
the tier order moved the SciQ gap from +0.342 to +0.639 and flipped OBQA from −0.228 to +0.408,
because the **last-emitted tier** is the one pushed off the consensus. If Qwen2.5-7B shows the
same order dependence, it is a property of the prompting scheme; if not, it is a Qwen3-4B quirk.

**What to run.** Upload the repo (§1.3), then:

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

**What to run.** Repo upload (§1.3), then:

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

## 6. Conventions for returned files

- **Naming mirrors the Qwen3-4B artefact exactly**, substituting `qwen2.5_7b` for `qwen3_4b`, so
  the two are trivially diffable (`kda_qwen2.5_7b_results.json` ↔ `kda_qwen3_4b_results.json`).
- Keep the `summary` block the scripts emit — the comparison analyses read it directly.
- Include the run log; the timing and VRAM lines belong in the comparison report.
- Record the **exact model revision** (`model.config._name_or_path` and the HF commit hash if
  pinned) in the summary, so a later re-run is identifiable.

---

## 7. Changelog

| Round | Added |
|---|---|
| 2026-09-04 | File created. Entries 1–4 populated from an exhaustive audit of every LLM-invoking script (§0.1). Entry 3 flagged as the critical path blocking the full free-response run. |

**When appending:** add a row to §0's status board, a full entry section, the comparison analysis,
and a changelog row. New Qwen3-4B work is not complete until its Qwen2.5 counterpart is queued
here.
