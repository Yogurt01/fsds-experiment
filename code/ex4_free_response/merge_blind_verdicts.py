"""Merge the annotator's verdicts from the blind copy back into the judge-validation sheet.

Counterpart to `build_blind_sheet.py`. That script strips the judge's columns so the human cannot
see them; this one carries the finished `human_verdict` column back, joining on `pair_key` --
the derivation-stable key, not row position -- and writes back **only** that column.

Safety properties, all asserted rather than assumed:

    * every pair_key in the blind sheet matches exactly one row of the target, and vice versa
    * every verdict parses under compute_kappa.py's alias table; a single unparseable or blank
      value aborts the merge rather than silently dropping a row from kappa
    * the target's other columns and its row order are compared before and after; any change
      outside `human_verdict` aborts
    * a refusal if the target already carries verdicts, unless --force

Usage:
    uv run --active python code/ex4_free_response/merge_blind_verdicts.py
    uv run --active python code/ex4_free_response/merge_blind_verdicts.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex4_free_response.build_blind_sheet import KEY_FIELDS, KEY_LENGTH, KEY_SEPARATOR
from ex4_free_response.compute_kappa import normalise_verdict
from utils.paths import resolve

TARGET = "results/ex4_free_response/validation_sheet_pilot_bidir.csv"
BLIND = "results/ex4_free_response/validation_sheet_pilot_bidir_BLIND.csv"
VERDICT_COLUMN = "human_verdict"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_file(path: str):
    """Return (comment lines, DictReader rows, fieldnames), preserving the `#` preamble."""
    with open(resolve(path), encoding="utf-8") as handle:
        raw = handle.readlines()
    comments = [line for line in raw if line.startswith("#")]
    body = [line for line in raw if not line.startswith("#")]
    reader = csv.DictReader(body)
    return comments, list(reader), list(reader.fieldnames or [])


def pair_key(row: Dict[str, str]) -> str:
    raw = KEY_SEPARATOR.join(str(row[field]) for field in KEY_FIELDS)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:KEY_LENGTH]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge blind verdicts back into the sheet.")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--blind", default=BLIND)
    parser.add_argument("--dry-run", action="store_true", help="Verify and report; write nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Merge even if the target already carries verdicts.")
    args = parser.parse_args(argv)

    target_path = resolve(args.target)
    hash_before = sha256_file(target_path)
    comments, target_rows, fieldnames = split_file(args.target)
    _, blind_rows, _ = split_file(args.blind)

    # --- join integrity -------------------------------------------------------------------
    target_keys = [pair_key(row) for row in target_rows]
    blind_keys = [row["pair_key"] for row in blind_rows]
    if len(set(target_keys)) != len(target_keys):
        print("  ERROR: pair_key is not unique in the target.")
        return 1
    if len(set(blind_keys)) != len(blind_keys):
        print("  ERROR: pair_key is not unique in the blind sheet.")
        return 1
    if set(target_keys) != set(blind_keys):
        print(f"  ERROR: key sets differ. "
              f"only-in-target={len(set(target_keys) - set(blind_keys))} "
              f"only-in-blind={len(set(blind_keys) - set(target_keys))}")
        return 1

    # --- verdict legality -----------------------------------------------------------------
    verdicts: Dict[str, str] = {}
    problems: List[str] = []
    for row in blind_rows:
        parsed = normalise_verdict(row.get(VERDICT_COLUMN))
        if parsed is None:
            problems.append(f"row_id {row.get('row_id')}: {row.get(VERDICT_COLUMN)!r}")
        else:
            verdicts[row["pair_key"]] = parsed
    if problems:
        print(f"  ERROR: {len(problems)} row(s) blank or unparseable; refusing a partial merge.")
        for problem in problems[:10]:
            print(f"    {problem}")
        return 1

    already = sum(1 for row in target_rows if (row.get(VERDICT_COLUMN) or "").strip())
    if already and not args.force:
        print(f"  REFUSING: the target already carries {already} verdict(s). Use --force to replace.")
        return 1

    # --- apply, in memory -----------------------------------------------------------------
    before = [dict(row) for row in target_rows]
    for key, row in zip(target_keys, target_rows):
        row[VERDICT_COLUMN] = verdicts[key]

    changed = {
        column: sum(1 for old, new in zip(before, target_rows) if old[column] != new[column])
        for column in fieldnames
    }
    unexpected = {c: n for c, n in changed.items() if n and c != VERDICT_COLUMN}
    if unexpected:
        print(f"  ERROR: columns other than {VERDICT_COLUMN} would change: {unexpected}")
        return 1

    print(f"  target rows            : {len(target_rows)}")
    print(f"  blind rows             : {len(blind_rows)}")
    print(f"  pair_key matched 1:1   : {len(set(target_keys) & set(blind_keys))}/{len(target_rows)}")
    print(f"  verdicts parsed        : {len(verdicts)}/{len(blind_rows)}")
    print(f"  cells changed          : {changed[VERDICT_COLUMN]} (all in {VERDICT_COLUMN})")
    print(f"  other columns changed  : none")
    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    with open(target_path, "w", newline="", encoding="utf-8") as handle:
        handle.writelines(comments)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(target_rows)

    print(f"  sha256 before          : {hash_before}")
    print(f"  sha256 after           : {sha256_file(target_path)}")
    print(f"  wrote {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
