"""Download the target LLM weights and tokenizers into `models/<name>/`, then verify them.

Two models are fetched by default:

    Qwen/Qwen3-4B-Instruct-2507  -> models/Qwen3-4B-Instruct-2507/   (local 4-bit eval)
    Qwen/Qwen2.5-7B-Instruct     -> models/Qwen2.5-7B-Instruct/      (cloud eval)

Both repositories are public, so no `HF_TOKEN` is required; one is still used if present
(`--token`, `$HF_TOKEN`, or a cached login) for higher rate limits.

Integrity verification
----------------------
`snapshot_download` already checks the ETag/size of every transferred file, but that check
is invisible after the fact and does not survive a partially-copied directory. This script
therefore re-verifies the materialised tree against the Hub's own metadata:

  * **LFS files** (the `*.safetensors` shards -- everything that actually matters) are
    verified by recomputing their **SHA-256** and comparing against `lfs.sha256` from the
    Hub API. This is a true content checksum.
  * **Regular files** (configs, tokenizer JSON) carry a git blob SHA-1 rather than a
    SHA-256, so they are verified by recomputing the **git blob SHA-1**
    (`sha1("blob <size>\\0" + bytes)`) and comparing against `blob_id`.
  * Every file is additionally checked for an exact **byte-size** match.

A non-zero exit status means at least one file is missing, truncated, or corrupt.

Usage:
    uv run --active python code/pre_data/download_models.py
    uv run --active python code/pre_data/download_models.py --verify-only
    uv run --active python code/pre_data/download_models.py --models Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

# --------------------------------------------------------------------------------------
# Cross-stage imports. This script lives in `code/<stage>/`, so `code/` itself is put on
# `sys.path`; `utils.paths` and `ex1_reproduce_KDA.kda_tiny` then resolve no matter which
# directory the script is launched from. The *project root* is deliberately NOT added --
# it contains a `datasets/` folder that would shadow the HuggingFace `datasets` package.
# --------------------------------------------------------------------------------------
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils.paths import MODELS_DIR, ensure_parent, resolve

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub", "fsspec")

# repo_id -> local subfolder name under models/
DEFAULT_MODELS = {
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
}

# Weights + tokenizer + config only. Keeps consolidated/duplicate formats off the disk.
IGNORE_PATTERNS = [
    "*.bin",          # superseded by *.safetensors in both repos
    "*.pth",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.gguf",
    "original/*",     # full-precision duplicates some Qwen repos ship
    ".gitattributes",
]

CHUNK = 1 << 22  # 4MB read buffer for hashing


def setup_logging(log_path: str, verbose: bool) -> logging.Logger:
    """DEBUG records to `log_path`, INFO records to stdout."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(ensure_parent(log_path), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("download_models")


def human(n_bytes: float) -> str:
    """Format a byte count for logs."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024.0:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f}PB"


def sha256_of(path: str) -> str:
    """SHA-256 of a file's contents (matches the Hub's `lfs.sha256`)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1_of(path: str) -> str:
    """Git blob SHA-1 of a file (matches the Hub's `blob_id` for non-LFS files).

    Git hashes `"blob <bytelength>\\0"` followed by the raw contents.
    """
    size = os.path.getsize(path)
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("utf-8"))
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_expected_files(api: HfApi, repo_id: str, token: Optional[str]) -> Dict[str, Dict]:
    """Map `filename -> {size, sha256, blob_id, is_lfs}` from the Hub's file metadata."""
    info = api.model_info(repo_id, files_metadata=True, token=token)
    expected: Dict[str, Dict] = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        sha256 = None
        if lfs is not None:
            # huggingface_hub exposes this as a dataclass in some versions, a dict in others.
            sha256 = getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)
        expected[sibling.rfilename] = {
            "size": getattr(sibling, "size", None),
            "sha256": sha256,
            "blob_id": getattr(sibling, "blob_id", None),
            "is_lfs": lfs is not None,
        }
    return expected


def matches_ignore(filename: str, patterns: Sequence[str]) -> bool:
    """True if `filename` matches any glob in `patterns` (same semantics as the Hub's)."""
    import fnmatch

    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def verify_directory(
    logger: logging.Logger,
    repo_id: str,
    local_dir: str,
    expected: Dict[str, Dict],
    ignore_patterns: Sequence[str],
) -> Dict:
    """Re-verify every downloaded file against the Hub metadata. Returns a report dict."""
    logger.info("Verifying %s", local_dir)

    checked, failures, skipped = [], [], []
    total_bytes = 0
    started = time.time()

    for filename, meta in sorted(expected.items()):
        if matches_ignore(filename, ignore_patterns):
            skipped.append({"file": filename, "reason": "ignored by download filter"})
            continue

        path = os.path.join(local_dir, filename)
        if not os.path.isfile(path):
            failures.append({"file": filename, "reason": "missing on disk"})
            logger.error("  MISSING   %s", filename)
            continue

        actual_size = os.path.getsize(path)
        total_bytes += actual_size

        if meta["size"] is not None and actual_size != meta["size"]:
            failures.append(
                {
                    "file": filename,
                    "reason": "size mismatch",
                    "expected": meta["size"],
                    "actual": actual_size,
                }
            )
            logger.error(
                "  BAD SIZE  %s (expected %s, got %s)", filename, meta["size"], actual_size
            )
            continue

        # Content checksum: SHA-256 for LFS weights, git blob SHA-1 for plain text files.
        if meta["sha256"]:
            algorithm, expected_digest = "sha256", meta["sha256"]
            actual_digest = sha256_of(path)
        elif meta["blob_id"]:
            algorithm, expected_digest = "git-sha1", meta["blob_id"]
            actual_digest = git_blob_sha1_of(path)
        else:
            skipped.append({"file": filename, "reason": "no checksum published by the Hub"})
            logger.warning("  NO HASH   %s (size %s verified only)", filename, human(actual_size))
            continue

        if actual_digest != expected_digest:
            failures.append(
                {
                    "file": filename,
                    "reason": f"{algorithm} mismatch",
                    "expected": expected_digest,
                    "actual": actual_digest,
                }
            )
            logger.error("  CORRUPT   %s (%s mismatch)", filename, algorithm)
            continue

        checked.append(
            {
                "file": filename,
                "size": actual_size,
                "algorithm": algorithm,
                "digest": actual_digest,
            }
        )
        logger.info(
            "  OK        %-46s %10s  %s:%s",
            filename,
            human(actual_size),
            algorithm,
            actual_digest[:16],
        )

    elapsed = time.time() - started
    logger.info(
        "  %d/%d files verified (%s) in %.1fs -- %d failed, %d skipped",
        len(checked),
        len(checked) + len(failures),
        human(total_bytes),
        elapsed,
        len(failures),
        len(skipped),
    )

    return {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "n_verified": len(checked),
        "n_failed": len(failures),
        "n_skipped": len(skipped),
        "total_bytes": total_bytes,
        "verify_seconds": round(elapsed, 2),
        "files": checked,
        "failures": failures,
        "skipped": skipped,
        "ok": not failures,
    }


def download_one(
    logger: logging.Logger,
    api: HfApi,
    repo_id: str,
    subdir: str,
    models_root: str,
    token: Optional[str],
    max_workers: int,
    force: bool,
    verify_only: bool,
) -> Dict:
    """Download (unless `verify_only`) and then verify a single repo."""
    local_dir = os.path.join(models_root, subdir)
    logger.info("=" * 88)
    logger.info("%s  ->  %s", repo_id, local_dir)
    logger.info("=" * 88)

    try:
        expected = fetch_expected_files(api, repo_id, token)
    except GatedRepoError:
        logger.error(
            "%s is gated. Accept its licence on the Hub and pass --token / set $HF_TOKEN.",
            repo_id,
        )
        return {"repo_id": repo_id, "local_dir": local_dir, "ok": False, "error": "gated repo"}
    except RepositoryNotFoundError:
        logger.error("%s does not exist on the Hub.", repo_id)
        return {"repo_id": repo_id, "local_dir": local_dir, "ok": False, "error": "repo not found"}

    wanted = {f: m for f, m in expected.items() if not matches_ignore(f, IGNORE_PATTERNS)}
    planned_bytes = sum(m["size"] or 0 for m in wanted.values())
    logger.info("Hub lists %d files; %d selected (%s)", len(expected), len(wanted), human(planned_bytes))

    download_seconds = 0.0
    if not verify_only:
        os.makedirs(local_dir, exist_ok=True)
        started = time.time()
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=IGNORE_PATTERNS,
            max_workers=max_workers,
            force_download=force,
            token=token,
        )
        download_seconds = time.time() - started
        logger.info(
            "Downloaded in %.1fs (%.1f MB/s average)",
            download_seconds,
            (planned_bytes / 1e6) / max(download_seconds, 1e-9),
        )
    else:
        logger.info("--verify-only: skipping transfer")

    report = verify_directory(logger, repo_id, local_dir, expected, IGNORE_PATTERNS)
    report["download_seconds"] = round(download_seconds, 2)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the target LLMs into models/ and verify their checksums.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run --active python code/pre_data/download_models.py\n"
            "  uv run --active python code/pre_data/download_models.py --verify-only\n"
            "  uv run --active python code/pre_data/download_models.py --models Qwen/Qwen2.5-7B-Instruct\n"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Hub repo ids to fetch (default: both Qwen targets).",
    )
    parser.add_argument(
        "--models-dir", default=MODELS_DIR, help="Destination root (default: <project>/models)."
    )
    parser.add_argument(
        "--report", default="results/ex1_reproduce_KDA_w_modernLLM/model_download_report.json", help="Verification report path."
    )
    parser.add_argument(
        "--log-file", default="results/ex1_reproduce_KDA_w_modernLLM/model_download.log", help="Execution log path."
    )
    parser.add_argument("--token", default=None, help="HF token (else $HF_TOKEN / cached login).")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers.")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    parser.add_argument(
        "--verify-only", action="store_true", help="Skip the transfer; only re-check what is on disk."
    )
    parser.add_argument("--verbose", action="store_true", help="Stream DEBUG logs to console.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(resolve(args.log_file), args.verbose)

    models_root = resolve(args.models_dir)
    os.makedirs(models_root, exist_ok=True)

    token = args.token or os.environ.get("HF_TOKEN") or None

    logger.info("Models root : %s", models_root)
    logger.info("Token       : %s", "provided" if token else "none (public repos)")
    logger.info("Requested   : %s", ", ".join(args.models))
    logger.info("")

    api = HfApi()
    reports: List[Dict] = []
    for repo_id in args.models:
        subdir = DEFAULT_MODELS.get(repo_id, repo_id.split("/")[-1])
        reports.append(
            download_one(
                logger=logger,
                api=api,
                repo_id=repo_id,
                subdir=subdir,
                models_root=models_root,
                token=token,
                max_workers=args.max_workers,
                force=args.force,
                verify_only=args.verify_only,
            )
        )
        logger.info("")

    payload = {
        "models_root": models_root,
        "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "all_ok": all(r.get("ok") for r in reports),
        "reports": reports,
    }
    report_path = ensure_parent(resolve(args.report))
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    logger.info("=" * 88)
    logger.info("SUMMARY")
    logger.info("=" * 88)
    for report in reports:
        status = "OK" if report.get("ok") else "FAILED"
        logger.info(
            "  %-8s %-32s %4d files  %10s  %s",
            status,
            report["repo_id"],
            report.get("n_verified", 0),
            human(report.get("total_bytes", 0)),
            report.get("error", ""),
        )
    logger.info("")
    logger.info("Report : %s", report_path)
    logger.info("Log    : %s", resolve(args.log_file))

    if not payload["all_ok"]:
        logger.error("Integrity verification FAILED -- see the report for the offending files.")
        return 1
    logger.info("All files present and checksum-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
