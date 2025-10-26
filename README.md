# Mixing Forum Analyzer

[![Tests](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test.yml/badge.svg)](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/steme855/mixing-forum-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/steme855/mixing-forum-analyzer)

> **Spart Audio Engineers 120 h/Jahr** durch semantische Suche in Mixing-Foren. Findet wiederkehrende Probleme (Kick-Resonanzen, Vocal-Sibilance) in <1 Sekunde.

## 💰 Business Value

**Problem:** Audio Engineers verbringen 2–3 h pro Woche mit der Recherche in Foren, um bekannte Mixing-Probleme erneut zu lösen.

**Lösung:** Die semantische Suche findet ähnliche Cases sofort. SBERT/TF-IDF Matching + Preset-Recommender liefern direkte Handlungsempfehlungen.

**Impact:**
- Zeitersparnis: ~120 h/Jahr ≙ ca. 3.600 € bei 30 €/h Freelance-Rate
- Wissensdatenbank: kuratierte Forenposts + Preset-Library für wiederkehrende Symptome
- Monitoring: Evaluation Notebook + Metriken-Dashboard für kontinuierliche Verbesserung

**Demo:** [Live-App (lokal via `streamlit run app.py`)] | **Impact-View:** Notebook-Export `docs/metrics_for_readme.md`

## 🎥 Demo

![Demo](docs/demo.gif)

> Aufnahme-Idee: QuickTime/OBS, Query “Kick klingt blechern, zu viel 3kHz”, Top-3 Ergebnisse zeigen, Preset-Empfehlung markieren.

## 📊 Performance

| Metrik | Wert | Benchmark |
|--------|------|-----------|
| MRR | 1.00 | >0.70 |
| Top-1-Accuracy | 1.00 | >0.60 |
| Top-3-Accuracy | 1.00 | >0.80 |
| Top-5-Accuracy | 1.00 | >0.90 |

Quelle: `notebooks/02_evaluation.ipynb` (29 Queries, TF-IDF Baseline). Markdown-Export siehe `docs/metrics_for_readme.md`.

## 🔍 Evaluation Workflow

1. `notebooks/02_evaluation.ipynb` enthält 29 Test-Queries mit Ground-Truth-Mapping.
2. `evaluation/metrics.py` liefert MRR, Top-K Accuracy, Cosine-Verteilungen, Latenz-Stats.
3. Confusion-Analyse (False Positives/Negatives) wird als DataFrame erzeugt.
4. Ergebnisse können bei Bedarf in ein Dashboard integriert werden (`docs/metrics_for_readme.md`).

## 🎛️ Preset Intelligence

- Symptom-Keywords → Frequenzbereiche (z. B. “blechern” → Cut @ 3 – 5 kHz)
- Severity-Scoring passt Gain-Empfehlungen an (light/medium/strong)
- Multi-Label Output: mehrere Presets pro Problem möglich (`presets/preset_recommender.py`)

## ⚙️ Tech Stack

- Streamlit UI mit SBERT/TF-IDF Hybrid (Lazy Load, TF-IDF Fallback)
- spaCy DE (robuste Fallback-Logik, Diagnostics in der App)
- RapidFuzz für Keyword-Boosting & Synonym-Handling
- Evaluation: Pandas, NumPy, eigene Metrics-Library (`evaluation/metrics.py`)

## 🚀 Quickstart (5 Minuten)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
streamlit run app.py
```

## 🗂️ Daten-Pipeline

Siehe `data/README.md` für Quellen, Preprocessing und Lizenzhinweise. Rohdaten → `data/raw/`, bereinigte Texte → `data/processed/`, Embeddings → `data/embeddings/`.

## 🧪 Tests & CI

- Unit-, Integration- und Smoke-Tests (`tests/`) decken Search, Presets und Streamlit-Import ab (15+ Checks).
- Test-Workflow (`.github/workflows/test.yml`): Python 3.9/3.10/3.11 Matrix, flake8, black, isort, Pytest mit Coverage, Upload zu Codecov.
- Deploy-Workflow (`.github/workflows/deploy.yml`): Automatisches HuggingFace-Deploy bei Änderungen an `app/`, `requirements.txt` oder README inkl. Health-Check.
- `pytest.ini`, `pyproject.toml`, `Makefile`, `requirements-dev.txt` bündeln Enterprise-Readiness (Linting, Formatting, Docker, TDD-Loop).
- Setup-Guide mit Secret-Handling & Badges: `docs/setup_guide.md`.
