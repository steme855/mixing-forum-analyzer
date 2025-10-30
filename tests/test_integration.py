from __future__ import annotations

import importlib
import importlib.util
import json
import time
from pathlib import Path

import pytest

from presets.preset_recommender import PresetRecommender
from preset_advisor.search import SemanticSearchEngine


@pytest.fixture(scope="module")
def engine() -> SemanticSearchEngine:
    corpus_path = Path("data/sample_corpus.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    return SemanticSearchEngine(corpus)


@pytest.mark.integration
def test_end_to_end_search_to_preset(engine: SemanticSearchEngine) -> None:
    query = "Kick ist zu laut im Mix"
    search_results = engine.query(query, top_k=5)
    assert search_results

    recommender = PresetRecommender()
    presets = recommender.suggest(query)
    assert presets
    assert presets[0].notes


@pytest.mark.integration
def test_multi_query_performance(engine: SemanticSearchEngine) -> None:
    queries = [
        "Kick zu laut",
        "Snare Gate chattet",
        "Vocals nasal bei 800 Hz",
        "Synth verdeckt Vocals",
        "Delay verschmiert",
    ]
    start = time.perf_counter()
    for query in queries:
        engine.query(query, top_k=5)
    avg_ms = (time.perf_counter() - start) * 1000 / len(queries)
    assert avg_ms < 100.0


@pytest.mark.integration
def test_streamlit_app_importable() -> None:
    module_name = "app.main"
    module = None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            module_name, Path("src/app/main.py")
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    assert module is not None
