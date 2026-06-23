"""Forum data ingestion helpers for the Mixing Forum Analyzer.

This package prepares a scalable knowledge-base layer that can ingest
sample corpora, processed exports, and future forum crawls from JSON,
JSONL, and CSV files.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from preset_advisor.core import normalize_text

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".ndjson", ".csv"}
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass
class ForumDocument:
    """Normalized forum record used by the search and analytics stack."""

    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    url: str = ""
    author: str = ""
    created_at: str = ""
    language: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        """Return the text that should be indexed by search backends."""

        parts = [self.title.strip(), self.text.strip()]
        return " ".join(part for part in parts if part).strip()

    def to_record(self) -> dict[str, Any]:
        """Serialize the document to a JSON-friendly record."""

        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "author": self.author,
            "created_at": self.created_at,
            "language": self.language,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass
class IngestionReport:
    """Summary of the loaded corpus and its provenance."""

    source_files: list[str] = field(default_factory=list)
    documents_loaded: int = 0
    documents_deduplicated: int = 0
    documents_by_source: dict[str, int] = field(default_factory=dict)
    documents_by_format: dict[str, int] = field(default_factory=dict)
    keyphrase_rows: int = 0


@dataclass
class IngestionResult:
    """Container returned by the ingestion pipeline."""

    documents: list[ForumDocument]
    report: IngestionReport


def _is_text_payload(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _coerce_tags(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _extract_text(payload: dict[str, Any]) -> str:
    for key in ("text", "body", "content", "message", "post"):
        value = payload.get(key)
        if _is_text_payload(value):
            return str(value).strip()
    return ""


def _extract_title(payload: dict[str, Any]) -> str:
    for key in ("title", "subject", "headline"):
        value = payload.get(key)
        if _is_text_payload(value):
            return str(value).strip()
    return ""


def _extract_doc_id(payload: dict[str, Any], fallback_prefix: str, index: int) -> str:
    for key in ("doc_id", "id", "post_id", "thread_id", "message_id"):
        value = payload.get(key)
        if _is_text_payload(value) or isinstance(value, (int, float)):
            return str(value).strip()
    return f"{fallback_prefix}_{index}"


def _read_json_records(path: Path) -> list[dict[str, Any] | str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        for key in ("documents", "records", "items", "posts"):
            value = payload.get(key)
            if isinstance(value, list):
                return list(value)
    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_jsonl_records(path: Path) -> Iterator[dict[str, Any] | str]:
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_keyphrase_index(data_root: Path) -> dict[str, list[str]]:
    keyphrase_path = data_root / "processed" / "keyphrases.csv"
    if not keyphrase_path.exists():
        return {}

    index: dict[str, list[str]] = {}
    with keyphrase_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            post_id = str(row.get("post_id", "")).strip()
            phrase = str(row.get("phrase", "")).strip()
            if not post_id or not phrase:
                continue
            index.setdefault(post_id, []).append(phrase)
    return index


def _document_from_payload(
    payload: dict[str, Any] | str,
    source: str,
    index: int,
    keyphrase_index: dict[str, list[str]] | None = None,
) -> ForumDocument | None:
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        return ForumDocument(
            doc_id=f"{Path(source).stem}_{index}",
            text=text,
            source=source,
            metadata={"source_format": "json_list"},
        )

    text = _extract_text(payload)
    if not text:
        return None

    title = _extract_title(payload)
    doc_id = _extract_doc_id(payload, Path(source).stem, index)
    tags = _coerce_tags(payload.get("tags"))
    metadata = {key: value for key, value in payload.items() if key not in {"text", "body", "content"}}

    if keyphrase_index and doc_id in keyphrase_index:
        metadata["keyphrases"] = list(dict.fromkeys(keyphrase_index[doc_id]))

    return ForumDocument(
        doc_id=doc_id,
        text=text,
        title=title,
        source=source,
        url=str(payload.get("url", "")).strip(),
        author=str(payload.get("author", "")).strip(),
        created_at=str(payload.get("created_at", payload.get("date", ""))).strip(),
        language=str(payload.get("language", "")).strip(),
        tags=tags,
        metadata=metadata,
    )


def _normalize_fingerprint(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"[^\wäöüß]+", " ", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def _iter_candidate_files(data_root: Path) -> list[Path]:
    candidates: list[Path] = []
    sample_corpus = data_root / "sample_corpus.json"
    if sample_corpus.exists():
        candidates.append(sample_corpus)

    processed_dir = data_root / "processed"
    raw_dir = data_root / "raw"

    for folder in (processed_dir, raw_dir):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                candidates.append(path)

    return candidates


def _load_documents_from_file(
    path: Path,
    root: Path,
    keyphrase_index: dict[str, list[str]] | None = None,
) -> list[ForumDocument]:
    suffix = path.suffix.lower()
    source = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

    if suffix in {".json", ".jsonl", ".ndjson"}:
        records = _read_json_records(path) if suffix == ".json" else list(_read_jsonl_records(path))
    elif suffix == ".csv":
        records = _read_csv_records(path)
    else:
        return []

    documents: list[ForumDocument] = []
    for index, record in enumerate(records):
        document = _document_from_payload(record, source=source, index=index, keyphrase_index=keyphrase_index)
        if document is not None:
            documents.append(document)
    return documents


def _merge_document(existing: ForumDocument, incoming: ForumDocument) -> ForumDocument:
    metadata = dict(existing.metadata)
    for key, value in incoming.metadata.items():
        if key == "keyphrases" and key in metadata:
            merged = list(dict.fromkeys([*metadata[key], *value]))
            metadata[key] = merged
            continue
        if key not in metadata:
            metadata[key] = value

    sources = list(dict.fromkeys([existing.source, incoming.source]))
    metadata["sources"] = sources

    tags = tuple(dict.fromkeys([*existing.tags, *incoming.tags]))
    title = existing.title or incoming.title
    author = existing.author or incoming.author
    created_at = existing.created_at or incoming.created_at
    language = existing.language or incoming.language
    url = existing.url or incoming.url

    return ForumDocument(
        doc_id=existing.doc_id,
        text=existing.text,
        title=title,
        source=existing.source,
        url=url,
        author=author,
        created_at=created_at,
        language=language,
        tags=tags,
        metadata=metadata,
    )


def load_forum_knowledge_base(data_root: Path | str | None = None) -> IngestionResult:
    """Load, normalize, and deduplicate the forum knowledge base."""

    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    keyphrase_index = _load_keyphrase_index(root)
    report = IngestionReport()
    documents: list[ForumDocument] = []
    seen: dict[str, int] = {}

    for path in _iter_candidate_files(root):
        report.source_files.append(str(path.relative_to(root)) if path.is_relative_to(root) else str(path))
        try:
            loaded = _load_documents_from_file(path, root=root, keyphrase_index=keyphrase_index)
        except Exception:
            continue

        report.documents_loaded += len(loaded)
        report.documents_by_format[path.suffix.lower().lstrip(".")] = (
            report.documents_by_format.get(path.suffix.lower().lstrip("."), 0) + len(loaded)
        )
        report.documents_by_source[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)] = len(loaded)

        for document in loaded:
            fingerprint = _normalize_fingerprint(document.search_text)
            if not fingerprint:
                continue
            existing_index = seen.get(fingerprint)
            if existing_index is None:
                seen[fingerprint] = len(documents)
                documents.append(document)
                continue
            documents[existing_index] = _merge_document(documents[existing_index], document)
            report.documents_deduplicated += 1

    report.keyphrase_rows = sum(len(values) for values in keyphrase_index.values())
    return IngestionResult(documents=documents, report=report)


def load_forum_corpus_texts(data_root: Path | str | None = None) -> list[str]:
    """Return the ingested corpus as plain texts for search backends."""

    result = load_forum_knowledge_base(data_root=data_root)
    texts = [document.search_text for document in result.documents if document.search_text]
    if texts:
        return texts
    return [
        "Die Snare ist zu laut und harsch.",
        "Kick verliert Punch, mehr Attack benoetigt.",
        "Vocals wirken nasal, 800 Hz leicht absenken.",
        "Bass maskiert die Kick, Sidechain Kompressor?",
        "S-Laute zu scharf, De-Esser einsetzen.",
    ]


def export_forum_documents(documents: Sequence[ForumDocument], out_path: Path | str) -> Path:
    """Export normalized documents as JSON for downstream processing."""

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = [document.to_record() for document in documents]
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_file
