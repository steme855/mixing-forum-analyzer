from __future__ import annotations

import json

from data_ingestion.crawlers.kvr import (
    crawl_kvr_urls,
    decide_kvr_urls,
    extract_forum_post,
    parse_robots_txt,
)


ROBOTS = """
User-agent: *
Disallow: /forum/search.php
Disallow: /forum/viewonline.php
Disallow: /mailman/
"""


def test_parse_robots_and_decide_kvr_urls() -> None:
    rules = parse_robots_txt(ROBOTS)
    decisions = decide_kvr_urls(
        [
            "https://www.kvraudio.com/forum/viewtopic.php?t=123",
            "https://www.kvraudio.com/forum/search.php?keywords=mix",
            "https://example.com/forum/viewtopic.php?t=123",
        ],
        ROBOTS,
    )

    assert rules["disallow"] == ["/forum/search.php", "/forum/viewonline.php", "/mailman/"]
    assert decisions[0].allowed is True
    assert decisions[1].allowed is False
    assert "robots" in decisions[1].reason
    assert decisions[2].allowed is False


def test_extract_forum_post_strips_short_or_empty_pages() -> None:
    html = """
    <html>
      <head><title>Mixing question - KVR Audio</title></head>
      <body>
        <script>ignore me</script>
        <p>This is a long public forum post about kick and bass masking in a mix.
        It describes low-end balance, sidechain compression, and EQ cleanup with
        enough text to pass the extraction threshold for the pilot crawler.</p>
      </body>
    </html>
    """

    post = extract_forum_post("https://www.kvraudio.com/forum/viewtopic.php?t=123", html)

    assert post is not None
    assert post.source == "kvr"
    assert post.title == "Mixing question"
    assert "kick and bass masking" in post.text
    assert post.metadata["license_status"] == "requires_source_terms_review"


def test_crawl_kvr_urls_dry_run_writes_manifest_only(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    processed_path = tmp_path / "processed.csv"
    manifest_path = tmp_path / "manifest.json"

    manifest = crawl_kvr_urls(
        [
            "https://www.kvraudio.com/forum/viewtopic.php?t=123",
            "https://www.kvraudio.com/forum/search.php?keywords=mix",
        ],
        out_raw_path=raw_path,
        out_processed_path=processed_path,
        manifest_path=manifest_path,
        dry_run=True,
        robots_text=ROBOTS,
    )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["dry_run"] is True
    assert saved["allowed_urls"] == 1
    assert saved["blocked_urls"] == 1
    assert raw_path.exists() is False
    assert processed_path.exists() is False
