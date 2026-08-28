"""Project-relative path helpers shared by every script in `code/`.

The workspace is laid out as:

    experiment/
      code/       <- these scripts
      datasets/   <- prepared, KDA-formatted datasets (sciq/, openbookqa/)
      results/    <- experiment outputs and execution logs
      question-score/

Nothing here hardcodes an absolute system path: `PROJECT_ROOT` is derived from this
file's own location, so the scripts run identically from any working directory and on
any machine.
"""

from __future__ import annotations

import os

# code/paths.py -> code/ -> experiment/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE_DIR = os.path.join(PROJECT_ROOT, "code")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

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
