🎧 Mixing Forum Analyzer
> **Semantic Search Engine for Mixing Forums – Preset Coaching in under 1 second**

<p align="center">
  <img src="assets/README-preview.png" alt="Mixing Forum Analyzer banner" width="100%">
</p>

---

### 🔖 Build Status

[![CI/CD](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_and_deploy.yml)
[![Tests](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_only.yml/badge.svg)](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_only.yml)
[![Codecov](https://codecov.io/gh/steme855/mixing-forum-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/steme855/mixing-forum-analyzer)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue?logo=python)](pyproject.toml)
[![License: MIT](https://img.shields.io/github/license/steme855/mixing-forum-analyzer.svg)](LICENSE)
[![Hugging Face Space](https://img.shields.io/badge/Space-online-brightgreen?logo=huggingface)](https://huggingface.co/spaces/steme855/mixing-forum-analyzer)

---

## 💡 Business Value

### 🎯 Problem
Audio engineers spend 2–3 hours per week searching forums for repeated mixing issues.

### ⚙️ Solution
A **semantic hybrid search** instantly retrieves similar cases (SBERT + TF-IDF) and the **Preset Coach** recommends precise EQ or gain adjustments.

### 📈 Impact

| 💼 Benefit | 💰 Result |
|-------------|-----------|
| ⏱ **Time Saved** | ~120 h/year ≈ € 3 600 (@ € 30 / h) |
| 📚 **Knowledge Base** | Curated forum posts + preset library |
| 📊 **Monitoring** | Evaluation notebook + metrics dashboard |

> 💻 **Demo:** `streamlit run app.py`  📊 **Impact View:** `docs/metrics_for_readme.md`

---

## 🎥 Demo Preview

![Demo](docs/demo.gif)

> Example query: “Kick klingt blechern, too much 3 kHz” → Top-3 matches + preset recommendation highlight.

---

## 🌟 Core Features

✅ **Hybrid Search:** SBERT (`MiniLM-L12-v2`) + TF-IDF fallback with RapidFuzz boost  
✅ **Preset Coach:** Adaptive EQ gain & Q-factors (`presets/preset_recommender.py`)  
✅ **Offline Mode:** TF-IDF fallback < 100 ms latency  
✅ **Production-Ready:** Streamlit UI · Health Checks · Auto-Deploy (Hugging Face)  
✅ **Evaluation Suite:** MRR · Top-K Accuracy · Latency metrics via `evaluation/metrics.py`

---

## 🔬 SBERT vs TF-IDF (Search Backends)

| Aspect | SBERT (Hybrid) | TF-IDF (Fallback) |
|:--|:--|:--|
| **Embedding** | `MiniLM-L12-v2 (384 d)` | Character + word n-grams |
| **Strengths** | Understands synonyms & phrasing | Deterministic · Fast |
| **Latency (Top-5)** | ≈ 65 ms CPU (cached) | ≈ 25 ms CPU |
| **Use Case** | Free-text queries / posts | Keywords / offline |
| **Scoring** | Cosine similarity + RapidFuzz | Cosine similarity (TF-IDF) |

---

## 📊 Evaluation Results

| Metric | ✅ Value | 🎯 Target |
|:--|:--:|:--:|
| **MRR** | 1.00 | > 0.70 |
| **Top-1 Accuracy** | 1.00 | > 0.60 |
| **Top-3 Accuracy** | 1.00 | > 0.80 |
| **Top-5 Accuracy** | 1.00 | > 0.90 |

_Source:_ `notebooks/02_evaluation.ipynb` (29 queries, TF-IDF baseline) → `docs/metrics_for_readme.md`

---

## 🔍 Evaluation Workflow

1️⃣ Run `notebooks/02_evaluation.ipynb` (29 queries + ground truth)  
2️⃣ `evaluation/metrics.py` → MRR, Top-K Accuracy, latency stats  
3️⃣ Confusion DataFrame for false positives / negatives  
4️⃣ Integrate results into dashboard (`docs/metrics_for_readme.md`)

---

## 🎛️ Preset Intelligence

- 🧩 **Symptom → Frequency:** “blechern” → cut 3 – 5 kHz  
- 📈 **Severity Scoring:** light / medium / strong gain adjustments  
- 🎚 **Multi-Label Output:** multiple presets per issue

---

## 🚀 Quickstart (5 min)

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
streamlit run app.py


⸻

🗂 Data Pipeline

raw/ → processed/ → embeddings/
Detailed in data/README.md (sources, preprocessing, license notes)

⸻

🧪 Tests & CI/CD

Stage	Tool	Description
🧩 Unit / Integration Tests	pytest, pytest-cov	Covers search + presets
🎨 Lint & Format	flake8, black, isort	via Makefile / pre-commit
🧰 CI Workflow	.github/workflows/test.yml	Python 3.9 → 3.11 + Codecov
🚀 Deploy Workflow	.github/workflows/deploy.yml	Auto Deploy → Hugging Face
🔐 Secrets Guide	docs/setup_guide.md	Badge & token management


⸻

📈 Quality Dashboard

Metric	Source	Purpose
🧪 Coverage	Codecov￼	Coverage & trend visuals
⚙️ Workflows	GitHub Actions￼	CI / Deploy runs
🤗 Deployment	Hugging Face Space￼	Latest build online


⸻

👤 Author

Stephan Meyer — Audio Engineer & Developer
🎓 SAE Certified  |  Evidence-based Audio Workflows  |  AI-Driven Automation
📫 GitHub Profile￼

⸻

⭐ If you like this project, give it a Star on GitHub — it helps others find open-source audio AI tools.
