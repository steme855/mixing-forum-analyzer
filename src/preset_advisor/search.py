"""Lightweight semantic search engine built on top of TF-IDF and optionally SBERT."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_ingestion import ForumDocument, load_forum_corpus_texts

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    text: str
    score: float


def _load_default_corpus() -> list[str]:
    texts = load_forum_corpus_texts()
    if texts:
        return texts
    logger.info("Kein Korpus gefunden – nutze eingebautes Fallback-Korpus.")
    return [
        "Die Snare ist zu laut und harsch.",
        "Kick verliert Punch, mehr Attack benoetigt.",
        "Vocals wirken nasal, 800 Hz leicht absenken.",
        "Bass maskiert die Kick, Sidechain Kompressor?",
        "S-Laute zu scharf, De-Esser einsetzen.",
    ]


class SemanticSearchEngine:
    """TF-IDF basierte Suche mit optionalem SBERT-Upgrade."""

    def __init__(self, documents: Optional[Sequence[Any]] = None) -> None:
        self.documents, self.doc_ids = self._prepare_documents(documents)
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.document_matrix = (
            self.vectorizer.fit_transform(self.documents) if self.documents else None
        )
        self.sbert_model: Optional[Any] = None  # SentenceTransformer | None
        self.sbert_embeddings: Optional[np.ndarray] = None
        self._sbert_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    @staticmethod
    def _prepare_documents(
        documents: Optional[Sequence[Any]],
    ) -> tuple[list[str], list[str]]:
        """Normalisiert das Eingabe-Korpus (Strings oder Dicts) in parallele Text/ID-Listen."""
        raw_documents = list(documents) if documents is not None else _load_default_corpus()
        texts: list[str] = []
        doc_ids: list[str] = []
        for idx, entry in enumerate(raw_documents):
            if isinstance(entry, str):
                text = entry.strip()
                if not text:
                    continue
                texts.append(text)
                doc_ids.append(f"doc_{idx}")
                continue
            if isinstance(entry, ForumDocument):
                text = entry.search_text.strip()
                if not text:
                    continue
                texts.append(text)
                doc_ids.append(entry.doc_id)
                continue
            if isinstance(entry, dict):
                text_value = entry.get("text") or entry.get("body") or entry.get("content")
                title_value = entry.get("title")
                if not isinstance(text_value, str):
                    continue
                text = text_value.strip()
                if not text:
                    continue
                if isinstance(title_value, str) and title_value.strip():
                    text = f"{title_value.strip()} {text}".strip()
                doc_id_value = entry.get("doc_id") or entry.get("id") or f"doc_{idx}"
                doc_ids.append(str(doc_id_value))
                texts.append(text)
                continue
            if hasattr(entry, "search_text") and hasattr(entry, "doc_id"):
                text = str(getattr(entry, "search_text")).strip()
                doc_id_value = str(getattr(entry, "doc_id")).strip()
                if text and doc_id_value:
                    texts.append(text)
                    doc_ids.append(doc_id_value)
        return texts, doc_ids

    def _ensure_sbert_embeddings(self) -> bool:
        """Lädt SBERT lazy – nur wenn tatsächlich benötigt."""
        if SentenceTransformer is None or not self.documents:
            return False
        if self.sbert_model is None:
            try:
                self.sbert_model = SentenceTransformer(self._sbert_model_name, device="cpu")
            except Exception as exc:
                logger.warning("SBERT konnte nicht geladen werden: %s", exc)
                self.sbert_model = None
                return False
        if self.sbert_embeddings is None:
            try:
                # encode() gibt seit sentence-transformers >=3.x ein np.ndarray zurück;
                # convert_to_numpy wurde entfernt – direkte Zuweisung ist korrekt.
                raw = self.sbert_model.encode(
                    self.documents,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                self.sbert_embeddings = np.asarray(raw, dtype=np.float32)
            except Exception as exc:
                logger.warning("SBERT-Embedding des Korpus fehlgeschlagen: %s", exc)
                self.sbert_embeddings = None
                return False
        return True

    def query(self, text: str, top_k: int = 5) -> list[SearchResult]:
        """TF-IDF Suche – rückwärtskompatible Schnittstelle."""
        return self.query_advanced(text, top_k=top_k, use_sbert=False)

    def query_advanced(
        self, text: str, top_k: int = 5, use_sbert: bool = False
    ) -> list[SearchResult]:
        """Suche mit SBERT (falls verfügbar und aktiviert), sonst TF-IDF."""
        if not text.strip() or not self.documents:
            return []

        if use_sbert and self._ensure_sbert_embeddings():
            try:
                raw_q = self.sbert_model.encode(
                    [text],
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                query_embedding = np.asarray(raw_q, dtype=np.float32)
                scores = cosine_similarity(query_embedding, self.sbert_embeddings)[0]
                top_indices = np.argsort(scores)[::-1][:top_k]
                return [
                    SearchResult(
                        doc_id=self.doc_ids[idx],
                        text=self.documents[idx],
                        score=float(scores[idx]),
                    )
                    for idx in top_indices
                ]
            except Exception as exc:
                logger.warning("SBERT-Query fehlgeschlagen, Fallback auf TF-IDF: %s", exc)

        # TF-IDF Fallback
        if self.document_matrix is None:
            return []
        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.document_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                doc_id=self.doc_ids[idx],
                text=self.documents[idx],
                score=float(scores[idx]),
            )
            for idx in top_indices
        ]

    def batch_query(
        self, queries: Iterable[str], top_k: int = 5
    ) -> list[list[SearchResult]]:
        """Batch-Query-Interface – rückwärtskompatibel."""
        return [self.query(query, top_k=top_k) for query in queries]

    def save_results(
        self,
        query: str,
        results: list[SearchResult],
        out_path: str = "data/processed/last_search.json",
    ) -> None:
        """Schreibt Query und Ergebnisse in eine JSON-Datei."""
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "query": query,
            "results": [
                {"doc_id": r.doc_id, "text": r.text, "score": r.score} for r in results
            ],
        }
        out_file.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Ergebnisse gespeichert: %s", out_file)
