---
title: 🎧 Mixing Forum Analyzer
emoji: 🎚️
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: true
license: mit
short_description: Semantic Search Engine for Mixing Forums
tags:
  - audio
  - semantic-search
  - machine-learning
  - streamlit
  - huggingface
  - nlp
thumbnail: https://cdn-uploads.huggingface.co/production/uploads/684f05d88dd67bb7d27a2287/N4PhKWhsz-kqiHZGf_VtU.png
sdk_version: 1.51.0
---

# 🎧 Mixing Forum Analyzer

> **Semantic Search Engine for Mixing Forums – Preset-Coaching in unter 1 Sekunde**

**[🚀 Live-Demo auf Hugging Face](https://huggingface.co/spaces/stepman/mixing-forum-analyzer)** | **[📂 Quellcode auf GitHub](https://github.com/steme855/mixing-forum-analyzer)**

[![Codecov](https://codecov.io/gh/steme855/mixing-forum-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/steme855/mixing-forum-analyzer)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue?logo=python)](pyproject.toml)
[![License: MIT](https://img.shields.io/github/license/steme855/mixing-forum-analyzer.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/HF%20Space-online-brightgreen?logo=huggingface)](https://huggingface.co/spaces/stepman/mixing-forum-analyzer)

---

<p align="center">
  <img src="docs/README-preview.png" alt="Mixing Forum Analyzer Banner" width="90%">
</p>

---

## 🎯 Warum dieses Tool?

Audio-Engineers verbringen 2–3 Stunden pro Woche damit, Foren nach wiederkehrenden Mixing-Problemen zu durchsuchen. Der **Mixing Forum Analyzer** löst das mit semantischer Hybridsuche (SBERT + TF-IDF) und einem integrierten **Preset-Coach**, der direkt EQ- und Gain-Empfehlungen liefert.

| 💼 Nutzen | 💰 Ergebnis |
|---|---|
| ⏱ **Zeitersparnis** | ≈ 120 h / Jahr ≈ € 3.600 (@ € 30 / h) |
| 📚 **Wissensbasis** | Kuratierte Forenbeiträge + Preset-Bibliothek |
| 📊 **Monitoring** | Evaluation-Notebook + Metriken-Dashboard |

---

## 🌟 Features

- ✅ **Hybridsuche:** SBERT (`MiniLM-L12-v2`) + TF-IDF-Fallback mit RapidFuzz-Boost
- ✅ **Preset-Coach:** Adaptive EQ-Gain & Q-Faktoren (`presets/preset_recommender.py`)
- ✅ **Offline-Modus:** TF-IDF-Fallback < 100 ms Latenz
- ✅ **Production-Ready:** Streamlit-UI · Health Checks · Auto-Deploy (Hugging Face)
- ✅ **Evaluation Suite:** MRR · Top-K Accuracy · Latenz via `evaluation/metrics.py`
- ✅ **Export:** Suchergebnisse als CSV oder JSON exportierbar

---

## 🚀 Quickstart

```bash
# 1. Repo klonen
git clone https://github.com/steme855/mixing-forum-analyzer.git
cd mixing-forum-analyzer

# 2. Virtuelle Umgebung
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt
python -m spacy download de_core_news_sm

# 4. App starten
streamlit run app.py
```

> **Hinweis:** SBERT wird beim ersten Start automatisch heruntergeladen (~90 MB).  
> Ohne Internetverbindung greift die App automatisch auf den TF-IDF-Fallback zurück.

---

## 📁 Projektstruktur

```
mixing-forum-analyzer/
├── app.py                  # Streamlit-Einstiegspunkt
├── api.py                  # REST-API-Einstiegspunkt
├── src/app/
│   └── main.py             # Kern-Logik (Suche, NLP, UI)
├── data/
│   ├── sample_corpus.json
│   ├── preset_map.json
│   ├── keyphrases.json
│   └── rule_tags.json
├── presets/
│   └── preset_recommender.py
├── evaluation/
│   └── metrics.py
├── notebooks/
│   └── 02_evaluation.ipynb
├── tests/
├── docs/
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## 🔧 Tech Stack

| Komponente | Technologie | Version |
|---|---|---|
| UI | Streamlit | ≥ 1.51 |
| Semantische Suche | sentence-transformers (SBERT) | ≥ 2.2 |
| NLP / Deutsch | spaCy (`de_core_news_sm`) | ≥ 3.7 |
| Fuzzy Matching | RapidFuzz | ≥ 3.0 |
| Fallback-Suche | scikit-learn (TF-IDF) | ≥ 1.3 |
| Laufzeit | Python | 3.9 – 3.11 |
| Deployment | Hugging Face Spaces | – |

---

## 🔬 SBERT vs. TF-IDF

| Aspekt | SBERT (Hybrid) | TF-IDF (Fallback) |
|:--|:--|:--|
| **Embedding** | `MiniLM-L12-v2 (384 d)` | Zeichen- + Wort-N-Gramme |
| **Stärken** | Versteht Synonyme & Formulierungen | Deterministisch · Schnell |
| **Latenz (Top-5)** | ≈ 65 ms CPU (gecacht) | ≈ 25 ms CPU |
| **Einsatzbereich** | Freitext-Anfragen | Stichworte / Offline |
| **Scoring** | Kosinus-Ähnlichkeit + RapidFuzz | Kosinus-Ähnlichkeit (TF-IDF) |

---

## 🎛️ Preset Intelligence

- 🧩 **Symptom → Frequenz:** „blechern“ → Cut bei 3–5 kHz
- 📈 **Schweregrad-Scoring:** leicht / mittel / stark (Gain-Anpassung)
- 🎚 **Multi-Label:** mehrere Presets pro Problem möglich

---

## 📊 Evaluation

| Metrik | ✅ Wert | 🎯 Ziel |
|:--|:--:|:--:|
| **MRR** | 1.00 | > 0.70 |
| **Top-1 Accuracy** | 1.00 | > 0.60 |
| **Top-3 Accuracy** | 1.00 | > 0.80 |
| **Top-5 Accuracy** | 1.00 | > 0.90 |

_Quelle:_ `notebooks/02_evaluation.ipynb` (29 Queries, TF-IDF Baseline) → `docs/metrics_for_readme.md`

**Workflow:**
1. `notebooks/02_evaluation.ipynb` ausführen (29 Queries + Ground Truth)
2. `evaluation/metrics.py` → MRR, Top-K Accuracy, Latenz
3. Confusion DataFrame für False Positives / Negatives
4. Ergebnisse ins Dashboard einpflegen (`docs/metrics_for_readme.md`)

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

Coverage-Report: [![Codecov](https://codecov.io/gh/steme855/mixing-forum-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/steme855/mixing-forum-analyzer)

---

## 🎥 Demo

![Demo](docs/demo.gif)

> Beispiel-Query: „Kick klingt blechern, too much 3 kHz“ → Top-3 Treffer + Preset-Empfehlung

---

<p align="center">
  <sub>Built with ❤️ and 🤖 AI-assisted development</sub>
</p>
