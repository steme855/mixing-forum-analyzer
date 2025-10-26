from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from preset_advisor.search import SearchResult, SemanticSearchEngine


@pytest.fixture(scope="module")
def engine() -> SemanticSearchEngine:
    corpus = json.loads(Path("data/sample_corpus.json").read_text(encoding="utf-8"))
    return SemanticSearchEngine(corpus)


@pytest.mark.unit
def test_semantic_search_returns_results(engine: SemanticSearchEngine) -> None:
    results = engine.query("Kick ist zu laut im Mix", top_k=5)
    assert len(results) == 5
    assert results[0].score > 0.5


@pytest.mark.unit
def test_search_result_structure(engine: SemanticSearchEngine) -> None:
    results = engine.query("Snare klingt harsch", top_k=3)
    assert results
    assert all(isinstance(result, SearchResult) for result in results)
    assert all(result.text for result in results)


@pytest.mark.unit
def test_search_scores_within_range(engine: SemanticSearchEngine) -> None:
    results = engine.query("Vocals nasal 800 Hz reduzieren", top_k=5)
    for result in results:
        assert 0.0 <= result.score <= 1.0


@pytest.mark.unit
def test_search_results_sorted_by_score(engine: SemanticSearchEngine) -> None:
    results = engine.query("Kick verliert Punch", top_k=5)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.unit
def test_search_respects_top_k(engine: SemanticSearchEngine) -> None:
    top_k = 3
    results = engine.query("S-Laute sind scharf", top_k=top_k)
    assert len(results) == top_k


@pytest.mark.unit
def test_search_latency_below_threshold(engine: SemanticSearchEngine) -> None:
    queries = [
        "Kick ist zu laut",
        "Snare Gate arbeitet schlecht",
        "Hi-Hats zischeln",
        "Synth verdeckt Vocals",
    ]
    start = time.perf_counter()
    for query in queries:
        engine.query(query, top_k=5)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 100.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,expected_ids",
    [
        ("Kick zu laut", {"doc_30", "doc_1"}),
        ("Snare braucht mehr Body", {"doc_22", "doc_0"}),
        ("Hi-Hats brauchen mehr Shimmer", {"doc_24", "doc_6"}),
    ],
)
def test_search_parametrized_queries(
    engine: SemanticSearchEngine, query: str, expected_ids: set[str]
) -> None:
    results = engine.query(query, top_k=5)
    assert any(result.doc_id in expected_ids for result in results)
