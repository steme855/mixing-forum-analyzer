#!/usr/bin/env python3
"""Run a compliance-first KVR pilot crawl.

Default behavior is dry-run. Use --execute only after reviewing the manifest
and confirming that the target URLs are allowed and appropriate for the project.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_ingestion.crawlers.kvr import crawl_kvr_urls  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compliance-first KVR pilot crawler")
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="KVR forum URL to evaluate or crawl. Can be passed multiple times.",
    )
    parser.add_argument(
        "--url-file",
        type=Path,
        help="Text file with one KVR forum URL per line.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually fetch allowed forum URLs. Without this flag, only a dry-run manifest is written.",
    )
    parser.add_argument("--max-threads", type=int, default=50)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--raw-out", type=Path, default=ROOT / "data" / "raw" / "kvr_pilot_crawl.jsonl")
    parser.add_argument(
        "--processed-out",
        type=Path,
        default=ROOT / "data" / "processed" / "kvr_pilot_corpus.csv",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "data" / "raw" / "kvr_pilot_manifest.json",
    )
    args = parser.parse_args()

    urls = list(args.url)
    if args.url_file:
        urls.extend(_load_urls(args.url_file))
    if not urls:
        parser.error("provide at least one --url or --url-file")

    manifest = crawl_kvr_urls(
        urls,
        out_raw_path=args.raw_out,
        out_processed_path=args.processed_out,
        manifest_path=args.manifest_out,
        dry_run=not args.execute,
        max_threads=max(1, args.max_threads),
        delay_seconds=max(0.5, args.delay_seconds),
    )

    mode = "execute" if args.execute else "dry-run"
    print(f"KVR pilot crawler finished ({mode}).")
    print(f"Manifest: {args.manifest_out}")
    print(f"Allowed URLs: {manifest['allowed_urls']}")
    print(f"Blocked URLs: {manifest['blocked_urls']}")
    print(f"Records written: {manifest['records_written']}")
    return 0


def _load_urls(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


if __name__ == "__main__":
    raise SystemExit(main())
