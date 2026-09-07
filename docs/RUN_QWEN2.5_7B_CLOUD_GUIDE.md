# Running the KDA Evaluation with `Qwen2.5-7B-Instruct` on Google Colab / Kaggle

A self-contained guide for reproducing this project's KDA evaluation with
**`Qwen/Qwen2.5-7B-Instruct`** on free or low-cost cloud GPUs. The 7B model does not fit
the local 4GB RTX 3050 used for [the Qwen3-4B run](kda_qwen3_4b_evaluation_report.md), so
it is run in the cloud instead.

Everything you need is in this document. The companion script
[`code/ex1_reproduce_KDA/run_qwen7b_eval.py`](../code/ex1_reproduce_KDA/run_qwen7b_eval.py)
is the same code as [§3](#3-execution-script) in file form — it imports nothing from this
repository and can be uploaded on its own.

---

## Contents

1. [Choose your platform](#1-choose-your-platform)
2. [Environment setup](#2-environment-setup)
3. [Execution script](#3-execution-script)
4. [Step-by-step: Google Colab](#4-step-by-step-google-colab)
5. [Step-by-step: Kaggle](#5-step-by-step-kaggle)
6. [Using this repository's prepared datasets](#6-using-this-repositorys-prepared-datasets)
7. [Outputs and how to download them](#7-outputs-and-how-to-download-them)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Choose your platform

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

## 2. Environment setup

### Cell 1 — install dependencies

```python
!pip install -q -U transformers accelerate bitsandbytes datasets pandas huggingface_hub
```

On Kaggle, pin `numpy` first if you hit a binary-compatibility error on import:

```python
!pip install -q -U "numpy<3" transformers accelerate bitsandbytes datasets pandas huggingface_hub
```

Confirm the versions actually loaded (Colab pre-installs older copies; a restart is
sometimes required after upgrading):

```python
import transformers, accelerate, bitsandbytes, datasets
print("transformers ", transformers.__version__)
print("accelerate   ", accelerate.__version__)
print("bitsandbytes ", bitsandbytes.__version__)
print("datasets     ", datasets.__version__)
```

> If `bitsandbytes` raises `CUDA Setup failed`, restart the runtime
> (`Runtime → Restart session`) and re-run this cell. The wheel needs to be imported
> against the CUDA runtime that is loaded at process start.

### Cell 2 — set `HF_TOKEN` (optional)

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

## 3. Execution script

Run **Cell 3** to write the evaluation script to disk. This is the complete, runnable
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

## 4. Step-by-step: Google Colab

1. **New notebook** → `Runtime → Change runtime type` → **T4 GPU** (or A100 on Pro+) → Save.
2. Run **Cell 1** (installs). If `bitsandbytes` complains, `Runtime → Restart session`, then
   re-run.
3. Run **Cell 2** (`HF_TOKEN`) — optional, but do add the secret if you have one.
4. Run **Cell 3** (`%%writefile run_qwen7b_eval.py`).
5. **Run the evaluation.** Free T4:

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --precision 8bit --batch-size 8 --out-dir /content/kda_out
   ```

   Colab Pro+ A100:

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --precision bf16 --batch-size 32 --out-dir /content/kda_out
   ```

   Or just let it choose:

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --out-dir /content/kda_out
   ```

6. **Smoke test first.** Before committing to a full run, confirm the whole path works on
   20 samples (~2 minutes including the model download):

   ```python
   !python run_qwen7b_eval.py --datasets sciq --limit 20 --batch-size 4 --out-dir /content/kda_smoke
   ```

7. **Download the results** — see [§7](#7-outputs-and-how-to-download-them).

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

## 5. Step-by-step: Kaggle

1. **New Notebook** → right panel → `Accelerator` → **GPU T4 x2** → `Internet` → **On**
   (required to download the model and datasets).
2. Run **Cell 1** (installs), then `Run → Restart & Clear Cell Outputs` if `bitsandbytes`
   fails to import, and re-run.
3. Run **Cell 2** with the Kaggle secrets snippet.
4. Run **Cell 3** (`%%writefile run_qwen7b_eval.py`).
5. **Run the evaluation.** With two T4s, `device_map="auto"` shards the model across both,
   so `fp16` fits without quantisation:

   ```python
   !python run_qwen7b_eval.py --datasets obqa sciq --precision fp16 --batch-size 16 --out-dir /kaggle/working/kda_out
   ```

6. Anything written under **`/kaggle/working/`** is saved as notebook output. Use
   `Save Version → Save & Run All (Commit)` for an unattended run; results appear under the
   version's **Output** tab when it finishes.

> **Kaggle quotas.** GPU time is capped at ~30 hours/week and a single session at 12 hours.
> The full 1,384-sample run uses well under an hour.

---

## 6. Using this repository's prepared datasets

By default the script rebuilds SciQ and OpenBookQA from the Hub, applying the same
formatting as [`code/pre_data/`](../code/pre_data/): drop SciQ rows with an empty
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
    --precision 8bit --batch-size 8 --out-dir /content/kda_out
```

The two files live at
[`datasets/sciq/sciq_test_full.json`](../datasets/sciq/sciq_test_full.json) (884 samples)
and [`datasets/openbookqa/obqa_test_full.json`](../datasets/openbookqa/obqa_test_full.json)
(500 samples).

> **Use `--local-data` if you intend to compare against the Qwen3-4B numbers.** The rebuilt
> datasets use the same seed and logic, but pinning the actual files removes any doubt
> about dataset drift on the Hub.

---

## 7. Outputs and how to download them

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

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` during scoring | Batch too large for the GPU | The script auto-halves the batch and retries. To avoid it entirely, lower `--batch-size` or drop to `--precision 4bit`. |
| `CUDA out of memory` while *loading* | Precision too high for the card | Use `8bit` or `4bit`. A single T4 cannot hold 7B in fp16 with room to run. |
| `bitsandbytes` → `CUDA Setup failed` | Wheel imported before the CUDA runtime settled | `Runtime → Restart session`, then re-run the install cell. |
| `ValueError: Cannot use bf16 ... device does not support` | T4 has no bfloat16 | Use `--precision fp16` / `8bit` / `4bit`, or leave `auto`. |
| Accuracy near 25% (chance) on all four options | Chat template not applied, or right padding | Confirm `tokenizer.padding_side = "left"` and that `apply_chat_template(..., add_generation_prompt=True)` ran. Both are handled by the script. |
| `Letter 'A' has no single-token spelling` | Non-Qwen tokenizer that splits bare letters | Score per-option sequence likelihood instead, or pick a tokenizer with single-token letters. |
| Model download is very slow | Unauthenticated rate limit | Set `HF_TOKEN` (§2, Cell 2). |
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

## Related documents

* [`kda_qwen3_4b_evaluation_report.md`](kda_qwen3_4b_evaluation_report.md) — the local
  4-bit Qwen3-4B run this guide's 7B numbers are meant to be compared against.
* [`kda_reproduction_summary.md`](kda_reproduction_summary.md) — the original
  `KDA_small` encoder-ensemble baseline on the same two datasets.
* [`KDA_Paper_Documentation.md`](KDA_Paper_Documentation.md) — the metric's definition and
  the limitations that motivate running it on a modern LLM.
* [`../README.md`](../README.md) — project overview and research framing.
