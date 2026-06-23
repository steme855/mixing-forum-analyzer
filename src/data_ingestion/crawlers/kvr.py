"""Compliance-first KVR forum pilot crawler.

The crawler is intentionally conservative:

- dry-run is the default in the CLI wrapper
- robots.txt is checked before fetching forum pages
- blocked or ambiguous URLs are reported, not fetched
- no login, cookies, search endpoints, or Cloudflare bypass logic
- usernames are not stored in exported records
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT_USER_AGENT = "MFAResearchBot/0.1 (+https://github.com/steme855/MFA)"
DEFAULT_ROBOTS_URL = "https://www.kvraudio.com/robots.txt"
DEFAULT_SOURCE = "kvr"
DEFAULT_MAX_THREADS = 50
DEFAULT_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class CrawlDecision:
    """Robots and scope decision for one URL."""

    url: str
    allowed: bool
    reason: str


@dataclass(frozen=True)
class CrawledForumPost:
    """Normalized record exported by the pilot crawler."""

    doc_id: str
    source: str
    url: str
    title: str
    text: str
    fetched_at: str
    language: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "fetched_at": self.fetched_at,
            "language": self.language,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


class _ForumHTMLExtractor(HTMLParser):
    """Small HTML text extractor tuned for forum pages."""

    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(html.unescape(data).split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
            return
        if len(text) >= 20:
            self.body_parts.append(text)


def fetch_text(url: str, user_agent: str = DEFAULT_USER_AGENT, timeout: int = 20) -> str:
    """Fetch a URL as text with an explicit research user agent."""

    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=timeout) as response:  # nosec B310 - explicit crawler target
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def parse_robots_txt(robots_text: str) -> dict[str, list[str]]:
    """Parse enough robots.txt for conservative allow/disallow decisions."""

    rules: dict[str, list[str]] = {"disallow": [], "allow": []}
    applies_to_all = False
    for raw_line in robots_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        key_lower = key.lower()
        if key_lower == "user-agent":
            applies_to_all = value == "*"
            continue
        if not applies_to_all:
            continue
        if key_lower == "disallow" and value:
            rules["disallow"].append(value)
        elif key_lower == "allow" and value:
            rules["allow"].append(value)
    return rules


def is_allowed_by_robots(url: str, rules: dict[str, list[str]]) -> CrawlDecision:
    """Return whether the URL is allowed by parsed robots rules."""

    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    allow_match = _longest_prefix(path, rules.get("allow", []))
    disallow_match = _longest_prefix(path, rules.get("disallow", []))

    if disallow_match and len(disallow_match) > len(allow_match):
        return CrawlDecision(url=url, allowed=False, reason=f"robots disallow: {disallow_match}")
    return CrawlDecision(url=url, allowed=True, reason="robots allow")


def decide_kvr_urls(urls: Iterable[str], robots_text: str) -> list[CrawlDecision]:
    """Validate URLs against KVR scope and robots rules."""

    rules = parse_robots_txt(robots_text)
    decisions: list[CrawlDecision] = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.netloc not in {"www.kvraudio.com", "kvraudio.com"}:
            decisions.append(CrawlDecision(url=url, allowed=False, reason="outside kvraudio.com"))
            continue
        if not parsed.path.startswith("/forum/"):
            decisions.append(CrawlDecision(url=url, allowed=False, reason="outside /forum/ scope"))
            continue
        decision = is_allowed_by_robots(url, rules)
        decisions.append(decision)
    return decisions


def extract_forum_post(url: str, html_text: str, fetched_at: str | None = None) -> CrawledForumPost | None:
    """Extract a normalized forum record from HTML."""

    extractor = _ForumHTMLExtractor()
    extractor.feed(html_text)

    title = _clean_title(" ".join(extractor.title_parts))
    text = _clean_text(" ".join(extractor.body_parts))
    if len(text) < 120:
        return None

    return CrawledForumPost(
        doc_id=_doc_id(url),
        source=DEFAULT_SOURCE,
        url=url,
        title=title,
        text=text[:4000],
        fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
        language="en",
        tags=("kvr", "forum", "external_crawl"),
        metadata={
            "license_status": "requires_source_terms_review",
            "crawler": "kvr_pilot",
            "text_truncated": len(text) > 4000,
        },
    )


def crawl_kvr_urls(
    urls: Iterable[str],
    out_raw_path: Path,
    out_processed_path: Path,
    manifest_path: Path,
    *,
    dry_run: bool = True,
    max_threads: int = DEFAULT_MAX_THREADS,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    fetcher: Callable[[str], str] | None = None,
    robots_text: str | None = None,
) -> dict[str, Any]:
    """Crawl allowed KVR URLs or write a dry-run compliance report."""

    url_list = list(dict.fromkeys(urls))[:max_threads]
    fetcher = fetcher or fetch_text
    robots_text = robots_text if robots_text is not None else fetcher(DEFAULT_ROBOTS_URL)
    decisions = decide_kvr_urls(url_list, robots_text)
    allowed_urls = [decision.url for decision in decisions if decision.allowed]
    blocked = [decision for decision in decisions if not decision.allowed]
    fetched_at = datetime.now(timezone.utc).isoformat()
    records: list[CrawledForumPost] = []

    if not dry_run:
        for index, url in enumerate(allowed_urls):
            html_text = fetcher(url)
            record = extract_forum_post(url, html_text, fetched_at=fetched_at)
            if record is not None:
                records.append(record)
            if index < len(allowed_urls) - 1:
                time.sleep(delay_seconds)

    manifest = {
        "schema_version": "1.0",
        "source": DEFAULT_SOURCE,
        "dry_run": dry_run,
        "created_at": fetched_at,
        "requested_urls": len(url_list),
        "allowed_urls": len(allowed_urls),
        "blocked_urls": len(blocked),
        "records_written": len(records),
        "robots_url": DEFAULT_ROBOTS_URL,
        "user_agent": DEFAULT_USER_AGENT,
        "limits": {
            "max_threads": max_threads,
            "delay_seconds": delay_seconds,
        },
        "decisions": [decision.__dict__ for decision in decisions],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not dry_run:
        write_jsonl(records, out_raw_path)
        write_processed_csv(records, out_processed_path)

    return manifest


def write_jsonl(records: Iterable[CrawledForumPost], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_record(), ensure_ascii=False) + "\n")


def write_processed_csv(records: Iterable[CrawledForumPost], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["doc_id", "text", "title", "source", "url", "language", "tags", "license_status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "doc_id": record.doc_id,
                    "text": record.text,
                    "title": record.title,
                    "source": record.source,
                    "url": record.url,
                    "language": record.language,
                    "tags": "|".join(record.tags),
                    "license_status": record.metadata.get("license_status", ""),
                }
            )


def _longest_prefix(path: str, patterns: Iterable[str]) -> str:
    matches = [pattern for pattern in patterns if pattern and path.startswith(pattern)]
    return max(matches, key=len) if matches else ""


def _clean_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value.replace(" - KVR Audio", "").strip()


def _clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _doc_id(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"kvr_{digest}"
