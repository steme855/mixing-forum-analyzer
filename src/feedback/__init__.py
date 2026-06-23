"""User feedback storage for search and assistant responses."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_FEEDBACK_PATH = Path(__file__).resolve().parents[2] / "data" / "feedback" / "feedback.jsonl"


@dataclass(frozen=True)
class FeedbackEntry:
    """A single user feedback event."""

    query: str
    rating: int
    comment: str = ""
    result_doc_ids: tuple[str, ...] = ()
    feedback_type: str = "search"
    session_id: str = ""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["result_doc_ids"] = list(self.result_doc_ids)
        return record


class FeedbackStore:
    """Append-only JSONL feedback store.

    JSONL keeps the MVP simple, inspectable, and easy to migrate to a database
    when API monetization or user accounts are introduced.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_FEEDBACK_PATH

    def append(self, entry: FeedbackEntry) -> FeedbackEntry:
        if not entry.query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= entry.rating <= 5:
            raise ValueError("rating must be between 1 and 5")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_record(), ensure_ascii=False) + "\n")
        return entry

    def list_entries(self, limit: int = 100) -> list[FeedbackEntry]:
        if not self.path.exists():
            return []

        rows: list[FeedbackEntry] = []
        with self.path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    rows.append(_entry_from_record(record))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        return rows[-max(1, limit) :]

    def summary(self) -> dict[str, Any]:
        entries = self.list_entries(limit=10_000)
        if not entries:
            return {"count": 0, "average_rating": None, "by_type": {}}

        by_type: dict[str, int] = {}
        for entry in entries:
            by_type[entry.feedback_type] = by_type.get(entry.feedback_type, 0) + 1

        average_rating = sum(entry.rating for entry in entries) / len(entries)
        return {
            "count": len(entries),
            "average_rating": round(average_rating, 2),
            "by_type": by_type,
        }


def _entry_from_record(record: dict[str, Any]) -> FeedbackEntry:
    result_doc_ids = record.get("result_doc_ids", ())
    if isinstance(result_doc_ids, Iterable) and not isinstance(result_doc_ids, str):
        doc_ids = tuple(str(item) for item in result_doc_ids)
    else:
        doc_ids = ()

    return FeedbackEntry(
        entry_id=str(record.get("entry_id", "")) or str(uuid.uuid4()),
        created_at=str(record.get("created_at", "")) or datetime.now(timezone.utc).isoformat(),
        query=str(record.get("query", "")),
        rating=int(record.get("rating", 0)),
        comment=str(record.get("comment", "")),
        result_doc_ids=doc_ids,
        feedback_type=str(record.get("feedback_type", "search")),
        session_id=str(record.get("session_id", "")),
        metadata=record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {},
    )
