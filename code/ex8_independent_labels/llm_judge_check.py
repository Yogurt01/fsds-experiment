"""Supplementary LLM-judge pass over the same items the human annotated. NOT a second annotator.

WHAT THIS IS NOT
----------------
This is not a second human rater, it is not an inter-annotator reliability check, and its output
must never be fed to the kappa gate, averaged with the human labels, or substituted for them.
The pre-registered gate in `kappa_independent_labels.py` takes two HUMAN sheets and nothing here
touches it. Output filenames deliberately avoid "annotator2" so this can never be mistaken for a
human rater in later work.

This project has already measured what an unvalidated LLM judge does: Task A found Qwen3-4B
over-crediting 13:1 against a stricter human check (docs/ex5_failure_audit/annotation_final_conclusion.md).
High agreement here is weak supplementary evidence, not validation of the rubric or the labels.

ISOLATION
---------
Each item gets a fresh, stateless generation. The prompt contains exactly two things: the rubric
text lifted verbatim from docs/ex8_independent_labels/ANNOTATION_GUIDE_RQ2.md, and one item's
fields. The judge is told nothing about metrics, scores, Setting A/B/C, strata, the research
question, or any prior rating -- and no conversation state carries between items. The exact
prompt for every item is written into the output so the isolation claim is auditable rather
than asserted.

Verbatim rubric extraction: section 2 in full for Q1; section 3 for Q2 minus the subsection
"Why the separate pass matters", which is procedural advice to a human working across two days
and has no analogue for a stateless call. Both extracted texts are recorded in the output.

MODEL
-----
Local Qwen3-4B-Instruct-2507 under 4-bit NF4, loaded exactly as code/ex1_reproduce_KDA/kda_qwen_eval.py
loads it. api.anthropic.com returned HTTP 401 (no key in the environment), so the API path was
unavailable. Greedy decoding, so the pass is reproducible.

Usage:
    .venv/bin/python code/ex8_independent_labels/llm_judge_check.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import ensure_parent
from utils.paths import resolve as resolve_path

GUIDE = "docs/ex8_independent_labels/ANNOTATION_GUIDE_RQ2.md"
MODEL_DIR = "models/Qwen3-4B-Instruct-2507"
OUT_DIR = "results/ex8_independent_labels"
Q1_IN = "results/ex8_independent_labels/q1_answerability_sheet.csv"
Q2_IN = "results/ex8_independent_labels/q2_context_following_sheet.csv"
Q1_OUT = "q1_llm_judge_check.csv"
Q2_OUT = "q2_llm_judge_check.csv"
PROMPTS_OUT = "llm_judge_prompts.json"


def extract_rubric(text: str, start: str, stop: str, drop: Sequence[str] = ()) -> str:
    lines = text.splitlines()
    begin = next(i for i, l in enumerate(lines) if l.startswith(start))
    end = next(i for i, l in enumerate(lines) if i > begin and l.startswith(stop))
    block, skipping = [], False
    for line in lines[begin:end]:
        if any(line.startswith(d) for d in drop):
            skipping = True
            continue
        if skipping and line.startswith("### "):
            skipping = False
        if not skipping:
            block.append(line)
    return "\n".join(block).strip()


def build_prompts() -> Dict[str, str]:
    with open(resolve_path(GUIDE), encoding="utf-8") as handle:
        text = handle.read()
    return {
        "q1": extract_rubric(text, "## 2. Sheet 1", "## 3. Sheet 2"),
        "q2": extract_rubric(text, "## 3. Sheet 2", "## 4. Filling",
                             drop=("### Why the separate pass matters",)),
    }


Q1_TASK = ("\n\nHere is one item.\n\n"
           "Question: {question}\n"
           "Option a: {a}\nOption b: {b}\nOption c: {c}\nOption d: {d}\n"
           "Gold answer: {gold}\n"
           "Passage: {passage}\n\n"
           "Reply with a single digit, 1 to 4, and nothing else.")

Q2_TASK = ("\n\nHere is one item.\n\n"
           "Question: {question}\n"
           "Option a: {a}\nOption b: {b}\nOption c: {c}\nOption d: {d}\n"
           "Passage: {passage}\n\n"
           "Reply with exactly one of: a, b, c, d, none. Nothing else.")


def read_sheet(path: str) -> List[Dict]:
    body = [l for l in open(resolve_path(path), encoding="utf-8") if not l.startswith("#")]
    return list(csv.DictReader(body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="Judge only the first N items (smoke test).")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rubrics = build_prompts()
    model_path = resolve_path(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
        dtype=torch.float16,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    model.eval()

    def ask(prompt: str, max_new_tokens: int = 8) -> str:
        """One stateless generation. No history is carried between calls."""
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tokenizer(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    started = time.time()
    saved_prompts = {"model": MODEL_DIR, "quantisation": "4-bit NF4", "decoding": "greedy",
                     "rubric_q1_verbatim": rubrics["q1"], "rubric_q2_verbatim": rubrics["q2"],
                     "task_template_q1": Q1_TASK, "task_template_q2": Q2_TASK,
                     "isolation": "one fresh generation per item; no state between calls; "
                                  "prompt contains only the rubric above and one item's fields",
                     "example_full_prompt_q1": None, "example_full_prompt_q2": None}

    # ---- Q1 ----
    rows = read_sheet(Q1_IN)
    if args.limit:
        rows = rows[:args.limit]
    q1_out = []
    for n, row in enumerate(rows, 1):
        prompt = rubrics["q1"] + Q1_TASK.format(
            question=row["question"], a=row["option_a"], b=row["option_b"],
            c=row["option_c"], d=row["option_d"], gold=row["gold_answer"],
            passage=row["passage"])
        if saved_prompts["example_full_prompt_q1"] is None:
            saved_prompts["example_full_prompt_q1"] = prompt
        raw = ask(prompt)
        match = re.search(r"[1-4]", raw)
        q1_out.append({"item_id": row["item_id"], "llm_rating": match.group(0) if match else "",
                       "raw_response": raw})
        if n % 20 == 0:
            print(f"  Q1 {n}/{len(rows)}  ({time.time()-started:.0f}s)")

    # ---- Q2 ----
    rows2 = read_sheet(Q2_IN)
    if args.limit:
        rows2 = rows2[:args.limit]
    q2_out = []
    for n, row in enumerate(rows2, 1):
        prompt = rubrics["q2"] + Q2_TASK.format(
            question=row["question"], a=row["option_a"], b=row["option_b"],
            c=row["option_c"], d=row["option_d"], passage=row["passage"])
        if saved_prompts["example_full_prompt_q2"] is None:
            saved_prompts["example_full_prompt_q2"] = prompt
        raw = ask(prompt)
        match = re.search(r"\b(a|b|c|d|none)\b", raw.lower())
        q2_out.append({"item_id": row["item_id"], "llm_stated_answer": match.group(1) if match else "",
                       "raw_response": raw})
        if n % 20 == 0:
            print(f"  Q2 {n}/{len(rows2)}  ({time.time()-started:.0f}s)")

    out_dir = resolve_path(args.out_dir)
    for name, data, cols in ((Q1_OUT, q1_out, ["item_id", "llm_rating", "raw_response"]),
                             (Q2_OUT, q2_out, ["item_id", "llm_stated_answer", "raw_response"])):
        path = ensure_parent(os.path.join(out_dir, name))
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("# SUPPLEMENTARY LLM-JUDGE OUTPUT -- NOT a human annotator, NOT a kappa gate input.\n")
            handle.write("# Qwen3-4B-Instruct-2507, 4-bit NF4, greedy, one stateless call per item.\n")
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            writer.writerows(data)
        print(f"wrote {os.path.relpath(path, resolve_path('.'))}  ({len(data)} rows)")

    path = ensure_parent(os.path.join(out_dir, PROMPTS_OUT))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(saved_prompts, handle, ensure_ascii=False, indent=1)
    print(f"wrote {os.path.relpath(path, resolve_path('.'))}")
    print(f"total {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
