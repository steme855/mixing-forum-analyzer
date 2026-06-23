from __future__ import annotations

import pytest

from feedback import FeedbackEntry, FeedbackStore


def test_feedback_store_appends_and_summarizes(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    entry = store.append(
        FeedbackEntry(
            query="Kick zu laut",
            rating=5,
            comment="Treffer passt",
            result_doc_ids=("doc_1", "doc_2"),
            feedback_type="search",
        )
    )

    entries = store.list_entries()
    summary = store.summary()

    assert entries[0].entry_id == entry.entry_id
    assert entries[0].result_doc_ids == ("doc_1", "doc_2")
    assert summary["count"] == 1
    assert summary["average_rating"] == 5.0
    assert summary["by_type"] == {"search": 1}


def test_feedback_store_rejects_invalid_rating(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    with pytest.raises(ValueError, match="rating"):
        store.append(FeedbackEntry(query="Snare", rating=6))


def test_feedback_store_rejects_empty_query(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    with pytest.raises(ValueError, match="query"):
        store.append(FeedbackEntry(query=" ", rating=3))
