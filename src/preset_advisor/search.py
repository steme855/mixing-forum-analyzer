"""Lightweight semantic search engine built on top of TF-IDF."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_CORPUS_PATH = Path("data/sample_corpus.json")


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    text: str
    score: float


def _load_default_corpus() -> list[str]:
    if DEFAULT_CORPUS_PATH.exists():
        try:
            data = json.loads(DEFAULT_CORPUS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return data
        except json.JSONDecodeError:
            pass
    return [
        "Die Snare ist zu laut und harsch.",
        "Kick verliert Punch, mehr Attack benoetigt.",
        "Vocals wirken nasal, 800 Hz leicht absenken.",
        "Bass maskiert die Kick, Sidechain Kompressor?",
        "S-Laute zu scharf, De-Esser einsetzen.",
    ]


class SemanticSearchEngine:
    """Simple TF-IDF based semantic search used for local evaluation."""

    def __init__(self, documents: Sequence[str] | None = None) -> None:
        self.documents = list(documents or _load_default_corpus())
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        self.document_matrix = (
            self.vectorizer.fit_transform(self.documents) if self.documents else None
        )

    def query(self, text: str, top_k: int = 5) -> list[SearchResult]:
        if not text.strip() or not self.documents:
            return []
        if self.document_matrix is None:
            return []
        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.document_matrix)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[SearchResult] = []
        for idx in top_indices:
            results.append(
                SearchResult(
                    doc_id=f"doc_{idx}",
                    text=self.documents[idx],
                    score=float(scores[idx]),
                )
            )
        return results

    def batch_query(self, queries: Iterable[str], top_k: int = 5) -> list[list[SearchResult]]:
        return [self.query(query, top_k=top_k) for query in queries]
