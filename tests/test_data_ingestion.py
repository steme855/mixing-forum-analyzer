from __future__ import annotations

import csv
import json
from pathlib import Path

from data_ingestion import ForumDocument, load_forum_corpus_texts, load_forum_knowledge_base


def test_load_forum_knowledge_base_uses_project_data() -> None:
    result = load_forum_knowledge_base(Path("data"))

    assert result.documents
    assert result.report.source_files
    assert result.report.documents_loaded >= len(result.documents)
    assert any("sample_corpus.json" in source for source in result.report.source_files)
    assert any("clusters.csv" in source for source in result.report.source_files)
    assert all(isinstance(document, ForumDocument) for document in result.documents)


def test_loader_supports_json_csv_and_jsonl(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    (data_root / "sample_corpus.json").write_text(
        json.dumps(["Kick zu laut", "Snare braucht mehr Body"], ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = processed_dir / "forum.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["post_id", "title", "text", "cluster"])
        writer.writeheader()
        writer.writerow(
            {
                "post_id": "csv-1",
                "title": "Vocals",
                "text": "sitzen zu weit hinten",
                "cluster": "1",
            }
        )

    jsonl_path = raw_dir / "forum.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": "jsonl-1",
                        "title": "Hi-Hats",
                        "text": "zu scharf",
                        "source": "forum",
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        encoding="utf-8",
    )

    result = load_forum_knowledge_base(data_root)
    texts = load_forum_corpus_texts(data_root)

    assert len(result.documents) == 4
    assert any(document.doc_id == "csv-1" for document in result.documents)
    assert any(document.doc_id == "jsonl-1" for document in result.documents)
    assert any(document.search_text == "Vocals sitzen zu weit hinten" for document in result.documents)
    assert texts == [document.search_text for document in result.documents]


def test_loader_deduplicates_repeated_text(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    processed_dir = data_root / "processed"
    processed_dir.mkdir(parents=True)

    (data_root / "sample_corpus.json").write_text(
        json.dumps(["Kick zu laut", "Kick zu laut"], ensure_ascii=False),
        encoding="utf-8",
    )

    clusters_path = processed_dir / "clusters.csv"
    with clusters_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["post_id", "text", "cluster"])
        writer.writeheader()
        writer.writerow({"post_id": "cluster-1", "text": "Kick zu laut", "cluster": "1"})

    keyphrases_path = processed_dir / "keyphrases.csv"
    with keyphrases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["post_id", "phrase"])
        writer.writeheader()
        writer.writerow({"post_id": "cluster-1", "phrase": "Kick"})
        writer.writerow({"post_id": "cluster-1", "phrase": "zu laut"})

    result = load_forum_knowledge_base(data_root)

    assert len(result.documents) == 1
    assert result.report.documents_deduplicated >= 2
    assert result.documents[0].metadata["sources"]
    assert "keyphrases" in result.documents[0].metadata
