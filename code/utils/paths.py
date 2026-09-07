"""Project-relative path helpers shared by every script under `code/`.

The workspace is laid out as:

    experiment/
      code/
        utils/              <- this module (shared helpers)
        pre_data/           <- dataset preparation scripts
        ex1_reproduce_KDA/  <- KDA baseline reproduction
        ex2_counterfactual/ <- counterfactual / disentanglement experiments
      datasets/             <- prepared, KDA-formatted datasets (sciq/, openbookqa/)
      results/              <- experiment outputs and execution logs, grouped by stage:
        reproduce_KDA_pipeline/     baseline KDA_small / KDA_tiny runs + dataset prep logs
        reproduce_KDA_w_modernLLM/  modern-LLM (Qwen) evaluations + model download reports
        category_questions/         question categorisations (basic_ / complicated_)
        counterfact_results/        counterfactual perturbation experiment
      models/               <- downloaded LLM weights + tokenizers
      question-score/       <- the reference KDA implementation (untouched)

Nothing here hardcodes an absolute system path: `PROJECT_ROOT` is derived from this
file's own location, so the scripts run identically from any working directory and on
any machine.
"""

from __future__ import annotations

import os

# code/utils/paths.py -> code/utils/ -> code/ -> experiment/
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(UTILS_DIR)
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# Experiment stages (kept here so no script has to re-derive them).
PRE_DATA_DIR = os.path.join(CODE_DIR, "pre_data")
EX1_DIR = os.path.join(CODE_DIR, "ex1_reproduce_KDA")
EX2_DIR = os.path.join(CODE_DIR, "ex2_counterfactual")

DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
QUESTION_SCORE_DIR = os.path.join(PROJECT_ROOT, "question-score")

SCIQ_DIR = os.path.join(DATASETS_DIR, "sciq")
OPENBOOKQA_DIR = os.path.join(DATASETS_DIR, "openbookqa")


def resolve(path: str, base: str = PROJECT_ROOT) -> str:
    """Interpret `path` relative to `base` (the project root by default).

    Absolute paths are returned untouched, so every command-line override still works.
    """
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def ensure_parent(path: str) -> str:
    """Create the parent directory of `path` if it does not exist yet."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
