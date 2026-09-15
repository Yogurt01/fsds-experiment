"""Merge annotator 2's Task B verdicts + notes from the BLIND copy into the master sheet.

Task B was already scored and adjudicated on 2026-09-09 (docs/ex5_failure_audit/corpus_defect_audit.md
section 8); `kappa_corpus_defect.py` reads the BLIND file directly and never needed this merge.
This script exists only to make the master sheet `corpus_defect_sheet_sciq33.csv` carry the final
labels for archival completeness.

    Join       strictly on `item_id` -- 33 <-> 33, exact set match asserted, no positional join.
    Transfers  `verdict` and `note`. Every other column is compared row-by-row before and after;
               a single change outside those two columns aborts the write.
    Safety     timestamped backup of the master; the BLIND file, its LOCK and its manifest are
               never opened for writing; a provenance sidecar records all three checksums.

Known consequence, by design: this stales `source.sha256` in
`corpus_defect_sheet_sciq33_BLIND_manifest.json`. The sidecar
`corpus_defect_sheet_sciq33.MERGE.json` records the pre-merge master hash so the chain stays
auditable. Regenerating the BLIND sheet from the now-labelled master would NOT be blind -- the
blind workflow is finished, so this is documentation, not a live risk.

Usage:
    uv run --active python code/ex5_failure_audit/merge_blind_corpus_verdicts.py --dry-run
    uv run --active python code/ex5_failure_audit/merge_blind_corpus_verdicts.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from ex5_failure_audit.kappa_corpus_defect import normalise_verdict
from utils.paths import ensure_parent, resolve

MASTER = "results/ex5_failure_audit/corpus_defect_sheet_sciq33.csv"
BLIND = "results/ex5_failure_audit/corpus_defect_sheet_sciq33_BLIND.csv"
LOCK = "results/ex5_failure_audit/corpus_defect_sheet_sciq33_BLIND.LOCK.json"
SIDECAR = "results/ex5_failure_audit/corpus_defect_sheet_sciq33.MERGE.json"

TRANSFER_COLUMNS = ("verdict", "note")
JOIN_KEY = "item_id"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_file(path: str):
    """Return (comment lines, rows, fieldnames), preserving the leading `#` preamble."""
    with open(resolve(path), encoding="utf-8") as handle:
        raw = handle.readlines()
    comments = [line for line in raw if line.startswith("#")]
    body = [line for line in raw if not line.startswith("#")]
    reader = csv.DictReader(body)
    rows = list(reader)
    return comments, rows, list(reader.fieldnames or [])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Merge Task B blind verdicts into the master sheet.")
    parser.add_argument("--master", default=MASTER)
    parser.add_argument("--blind", default=BLIND)
    parser.add_argument("--dry-run", action="store_true", help="Verify and report; write nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Merge even if the master already carries verdicts.")
    args = parser.parse_args(argv)

    master_path = resolve(args.master)
    hash_before = sha256_file(master_path)
    blind_hash = sha256_file(resolve(args.blind))

    comments, master_rows, fieldnames = split_file(args.master)
    _, blind_rows, blind_fields = split_file(args.blind)

    for col in TRANSFER_COLUMNS:
        if col not in fieldnames:
            print(f"  ERROR: master has no `{col}` column ({fieldnames}).")
            return 1
        if col not in blind_fields:
            print(f"  ERROR: blind sheet has no `{col}` column ({blind_fields}).")
            return 1

    # --- join integrity: strict, on item_id, exact set match -----------------------------
    master_ids = [r[JOIN_KEY] for r in master_rows]
    blind_ids = [r[JOIN_KEY] for r in blind_rows]
    if len(set(master_ids)) != len(master_ids):
        print("  ERROR: item_id is not unique in the master.")
        return 1
    if len(set(blind_ids)) != len(blind_ids):
        print("  ERROR: item_id is not unique in the blind sheet.")
        return 1
    if set(master_ids) != set(blind_ids):
        only_m = sorted(set(master_ids) - set(blind_ids))
        only_b = sorted(set(blind_ids) - set(master_ids))
        print(f"  ERROR: key sets differ. only-in-master={only_m} only-in-blind={only_b}")
        return 1
    if len(master_rows) != 33:
        print(f"  ERROR: expected 33 master rows, found {len(master_rows)}.")
        return 1

    # --- verdict legality: every blind row must carry a parsable verdict -----------------
    transfer: Dict[str, Dict[str, str]] = {}
    problems: List[str] = []
    for row in blind_rows:
        parsed = normalise_verdict(row.get("verdict"))
        if parsed is None:
            problems.append(f"item_id {row.get(JOIN_KEY)}: {row.get('verdict')!r}")
        else:
            transfer[row[JOIN_KEY]] = {"verdict": parsed, "note": (row.get("note") or "").strip()}
    if problems:
        print(f"  ERROR: {len(problems)} blind row(s) blank or unparsable; refusing partial merge.")
        for p in problems[:10]:
            print(f"    {p}")
        return 1

    already = sum(1 for r in master_rows if (r.get("verdict") or "").strip())
    if already and not args.force:
        print(f"  REFUSING: the master already carries {already} verdict(s). Use --force to replace.")
        return 1

    # --- apply in memory, then diff every non-transfer column ---------------------------
    before = [dict(r) for r in master_rows]
    for row in master_rows:
        row["verdict"] = transfer[row[JOIN_KEY]]["verdict"]
        row["note"] = transfer[row[JOIN_KEY]]["note"]

    changed = {
        col: sum(1 for old, new in zip(before, master_rows) if old[col] != new[col])
        for col in fieldnames
    }
    unexpected = {c: n for c, n in changed.items() if n and c not in TRANSFER_COLUMNS}
    if unexpected:
        print(f"  ERROR: columns outside {TRANSFER_COLUMNS} would change: {unexpected}")
        return 1

    from collections import Counter
    dist = Counter(r["verdict"] for r in master_rows)

    print(f"  join key              : {JOIN_KEY} (strict; 33 <-> 33, exact set match)")
    print(f"  master rows           : {len(master_rows)}")
    print(f"  blind rows            : {len(blind_rows)}")
    print(f"  verdicts transferred  : {changed['verdict']}")
    print(f"  notes transferred     : {sum(1 for r in master_rows if r['note'])}")
    print(f"  other columns changed : none")
    print(f"  verdict distribution  : {dict(dist)}")

    if args.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    # --- backup, write, sidecar --------------------------------------------------------
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = f"{master_path}.pre_merge_{stamp}.bak"
    shutil.copy2(master_path, backup_path)

    with open(master_path, "w", newline="", encoding="utf-8") as handle:
        handle.writelines(comments)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(master_rows)

    hash_after = sha256_file(master_path)
    sidecar = {
        "_provenance": {
            "what": "Post-annotation merge of Task B blind verdicts into the master sheet.",
            "when": stamp,
            "why": "Archival only. Task B was scored and adjudicated 2026-09-09 "
                   "(docs/ex5_failure_audit/corpus_defect_audit.md section 8); the scorer reads the "
                   "BLIND file directly and this merge was never required by the protocol.",
            "generator": "code/ex5_failure_audit/merge_blind_corpus_verdicts.py",
            "join_key": JOIN_KEY,
            "columns_transferred": list(TRANSFER_COLUMNS),
            "columns_preserved": [c for c in fieldnames if c not in TRANSFER_COLUMNS],
        },
        "blind_source": {
            "path": args.blind,
            "sha256": blind_hash,
            "matches_lock": blind_hash == json.load(open(resolve(LOCK)))["sha256"],
        },
        "master": {
            "path": args.master,
            "sha256_before": hash_before,
            "sha256_after": hash_after,
            "backup": os.path.relpath(backup_path, resolve(".")),
        },
        "stale_reference_note": (
            "corpus_defect_sheet_sciq33_BLIND_manifest.json still records source.sha256 = "
            f"{hash_before} (the pre-merge master). That manifest is a historical record of the "
            "blind build and is intentionally left unchanged; this sidecar is the pointer to the "
            "post-merge state."
        ),
        "verdict_distribution": dict(dist),
        "adjudication_note": (
            "The raw human labels equal the adjudicated census: the only disagreement (item 584) "
            "was resolved to the human's ON_TOPIC in corpus_defect_adjudication.json, so the "
            "merged master reflects the final published result (12 MISMATCH / 21 ON_TOPIC)."
        ),
    }
    with open(ensure_parent(resolve(SIDECAR)), "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, ensure_ascii=False, indent=2)

    # --- verify the file on disk -----------------------------------------------------
    _, check_rows, _ = split_file(args.master)
    assert len(check_rows) == 33, "row count changed on disk"
    assert all(normalise_verdict(r["verdict"]) for r in check_rows), "unparsable verdict on disk"
    assert Counter(r["verdict"] for r in check_rows) == dist, "distribution changed on disk"

    print(f"  backup                : {backup_path}")
    print(f"  sha256 before / after : {hash_before[:16]} / {hash_after[:16]}")
    print(f"  wrote {master_path}")
    print(f"  wrote {resolve(SIDECAR)}")
    print(f"  BLIND file untouched  : {sha256_file(resolve(args.blind)) == blind_hash}")

    print("\n  first 3 updated rows:")
    for row in check_rows[:3]:
        note = (row["note"][:70] + "...") if len(row["note"]) > 70 else row["note"]
        print(f"    item_id={row['item_id']:>4}  verdict={row['verdict']:<9}  "
              f"screen={row.get('screen','?'):<10}  note={note!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
