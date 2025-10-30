"""Lightweight smoke tests to ensure the demo environment is intact."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


CRITICAL_MODULES = [
    "streamlit",
    "sentence_transformers",
    "torch",
    "spacy",
    "rapidfuzz",
    "sklearn",
]


def test_critical_imports() -> None:
    """Ensure all top-level dependencies are importable."""

    for module_name in CRITICAL_MODULES:
        importlib.import_module(module_name)


def test_torch_cpu_tensor_ops() -> None:
    """Verify basic tensor arithmetic works on the default device."""

    torch = importlib.import_module("torch")
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assert torch.isfinite(tensor).all()
    assert torch.allclose(tensor.sum(), torch.tensor(6.0))


def test_sentence_transformers_available() -> None:
    """Check that SentenceTransformer API is present and usable."""

    st_module = pytest.importorskip("sentence_transformers")
    assert hasattr(st_module, "SentenceTransformer")


def test_sample_corpus_json_structure() -> None:
    """Sample corpus should be valid JSON list of strings."""

    corpus_path = Path("data/sample_corpus.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    assert isinstance(corpus, list) and corpus, (
        "Corpus muss eine nicht-leere Liste sein"
    )
    assert all(isinstance(entry, str) for entry in corpus)


def test_preset_map_json_loadable() -> None:
    """Preset map JSON should parse even if empty."""

    preset_path = Path("data/preset_map.json")
    data = json.loads(preset_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
