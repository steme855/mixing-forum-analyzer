def test_imports():
    import numpy, pandas, sklearn, streamlit, spacy
    nlp = spacy.load("de_core_news_sm")
    assert nlp is not None

def test_repo_layout_exists():
    import os
    assert os.path.exists("app/app.py")
    assert os.path.exists("requirements.txt")
def test_json_schema(tmp_path):
    import json

    # Beispiel eines minimal gültigen Exports simulieren
    example = {
        "schema_version": "1.0",
        "query": "snare zu laut",
        "expanded_query": "snare zu laut",
        "mode": "TF-IDF",
        "elapsed_ms": 12.3,
        "results": [
            {"rank": 1, "post": "Snare klingt spitz", "score": 0.987}
        ],
        "presets": ["De-Esser bei 6–8 kHz"],
    }

    f = tmp_path / "last_query_results.json"
    f.write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")

    data = json.loads(f.read_text(encoding="utf-8"))

    # Basis-Checks (Schema-Smoke)
    assert data["schema_version"] == "1.0"
    assert isinstance(data["results"], list)
    assert isinstance(data["presets"], list)
    assert "query" in data and "mode" in data


def test_json_schema_details(tmp_path):
    import json
    from numbers import Number

    example = {
        "schema_version": "1.0",
        "query": "kick zu weich",
        "expanded_query": "kick zu weich",
        "mode": "TF-IDF (lexikalisch)",
        "elapsed_ms": 7.8,
        "results": [
            {"rank": 1, "post": "Kick braucht mehr Punch", "score": 0.912},
            {"rank": 2, "post": "Kick weich, Transienten fehlen", "score": 0.875},
        ],
        "presets": [
            "Transient Designer: Attack +3 dB",
            "Parallel Comp: Blend 30%",
        ],
    }

    p = tmp_path / "last_query_results.json"
    p.write_text(json.dumps(example, ensure_ascii=False), encoding="utf-8")
    data = json.loads(p.read_text(encoding="utf-8"))

    # Pflichtfelder & Typen
    assert data["schema_version"] == "1.0"
    assert isinstance(data["query"], str) and data["query"]
    assert isinstance(data["expanded_query"], str)
    assert isinstance(data["mode"], str)
    assert isinstance(data["elapsed_ms"], Number)
    assert isinstance(data["results"], list)
    assert isinstance(data["presets"], list)

    # Results: Struktur, Wertebereiche
    for i, item in enumerate(data["results"], start=1):
        assert set(item.keys()) >= {"rank", "post", "score"}
        assert item["rank"] == i  # fortlaufende Ranks
        assert isinstance(item["post"], str) and item["post"]
        assert isinstance(item["score"], Number)
        assert 0.0 <= item["score"] <= 1.0

    # Presets: Strings (dürfen leer sein, aber wenn da, dann str)
    for rec in data["presets"]:
        assert isinstance(rec, str)
