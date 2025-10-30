#!/usr/bin/env python3
"""Deploy the repository contents to a Hugging Face Space.

The script is intended to be executed from CI. It uploads the project to the
configured Space and performs an optional health check afterwards.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


try:
    from huggingface_hub import HfApi
except ImportError as exc:  # pragma: no cover - fail fast in CI
    raise SystemExit(
        "huggingface_hub is required. Install it with `pip install huggingface_hub[cli]`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOW_PATTERNS: tuple[str, ...] = (
    "app.py",
    "api.py",
    ".streamlit/**",
    "src/**",
    "presets/**",
    "data/sample_corpus.json",
    "requirements*.txt",
    "pyproject.toml",
    "sitecustomize.py",
    "README.md",
    "assets/**",
)

IGNORE_PATTERNS: tuple[str, ...] = (
    ".git/**",
    ".github/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    "docs/**",
    "notebooks/**",
    "tests/**",
    "**/__pycache__/**",
    "*.ipynb",
    "backup-*/**",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the current repository state to a Hugging Face Space."
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (defaults to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--space-id",
        default=os.environ.get("HF_SPACE_ID"),
        help="Target Space identifier like `username/space` (defaults to HF_SPACE_ID env var).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to upload (defaults to project root).",
    )
    parser.add_argument(
        "--health-url",
        default=os.environ.get("HF_HEALTH_URL"),
        help="Override health check URL (defaults to https://huggingface.co/spaces/<space_id>).",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the health check step after uploading to the Space.",
    )
    return parser.parse_args()


def ensure_inputs(token: str | None, space_id: str | None) -> tuple[str, str]:
    if not token:
        raise SystemExit("HF token missing. Provide --token or set HF_TOKEN secret.")
    if not space_id:
        raise SystemExit("Space id missing. Provide --space-id or set HF_SPACE_ID secret.")
    return token, space_id


def upload(
    api: HfApi,
    space_id: str,
    repo_root: Path,
    allow_patterns: Iterable[str],
    ignore_patterns: Iterable[str],
) -> None:
    commit_message = f"CI deploy ({datetime.now(timezone.utc).isoformat(timespec='seconds')})"
    api.upload_folder(
        folder_path=str(repo_root),
        repo_id=space_id,
        repo_type="space",
        path_in_repo=".",
        allow_patterns=list(allow_patterns),
        ignore_patterns=list(ignore_patterns),
        commit_message=commit_message,
    )
    logging.info("Uploaded repository to %s", space_id)


def health_check(url: str, timeout: int = 30) -> None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = response.status
            if status >= 400:
                raise RuntimeError(f"Health check returned status {status} for {url}")
            logging.info("Health check succeeded with status %s for %s", status, url)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Health check failed for {url}: {exc}") from exc


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    token, space_id = ensure_inputs(args.token, args.space_id)

    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"Repository root {repo_root} does not exist or is not a directory.")

    api = HfApi(token=token)
    upload(api, space_id, repo_root, ALLOW_PATTERNS, IGNORE_PATTERNS)

    if args.skip_health_check:
        logging.info("Skipping health check (disabled).")
        return 0

    health_url = args.health_url or f"https://huggingface.co/spaces/{space_id}"
    health_check(health_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
