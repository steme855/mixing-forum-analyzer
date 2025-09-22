---
title: "Mixing Forum Analyzer"
emoji: "🎚"
colorFrom: "red"
colorTo: "purple"
sdk: "streamlit"
sdk_version: "1.49.1"
app_file: "app/app.py"
pinned: false
license: "mit"
short_description: "KI-Demo: Mixing Forum Analyzer v0.1"
tags:
  - streamlit
  - nlp
  - audio
---

# 🎚 Mixing Forum Analyzer

> *KI-Demo: Automatische Analyse und Suche nach ähnlichen Mixing-Problemen in Foren-Posts (Kick, Snare, Vocals etc.)*  
> *Ziel: KI-gestützte Mixing-Support-Tools für Engineers, Education und Plugin-Hersteller.*

[![CI](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/steme855/mixing-forum-analyzer/actions)
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open%20Space-blue)](https://huggingface.co/spaces/Stepman/mixing-forum-analyzer)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Live-Demo

👉 Direkt ausprobieren im Hugging Face Space:  
https://huggingface.co/spaces/Stepman/mixing-forum-analyzer

![Demo GIF](app/assets/demo.gif)

---

## ⚡ Features

- **Semantische Suche mit SBERT** → ähnliche Mixing-Probleme in Foren finden  
- **Streamlit-App** → Texteingabe & Vorschläge in Echtzeit  
- **spaCy-Analyse** → Wortarten, Lemmata, Sound-Adjektive  
- **Preset-Map** → erste Mapping-Ideen basierend auf *Handbuch der Tonstudiotechnik*

---

## 🛠 Setup (lokal)

```bash
git clone https://github.com/steme855/mixing-forum-analyzer
cd mixing-forum-analyzer

# (Empfohlen) virtuelles Environment
python3 -m venv venv && source venv/bin/activate

# Abhängigkeiten
pip install -r requirements.txt

# (optional) deutsches spaCy-Modell
python -m spacy download de_core_news_sm

# Start
streamlit run app/app.py
```

---

## 🐞 Troubleshooting

- **SBERT lädt nicht / Triton-Warnungen (macOS/CPU):**  
  ```bash
  pip uninstall -y triton
  pip install --no-cache-dir "torch==2.2.2" sentence-transformers==3.0.1
  ```

- **spaCy DE-Modell fehlt (rechte Spalte leer):**
  ```bash
  python -m spacy download de_core_news_sm
  ```
  Alternativ (Direktlink):
  ```bash
  pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl
  ```

- **Fuzzy Matching ohne RapidFuzz (Fallback aktiv):**
  Wenn `rapidfuzz` nicht installiert ist, nutzt die App automatisch einen
  einfacheren Fuzzy-Vergleich auf Basis der Python-Standardbibliothek. Für
  präzisere Scores empfiehlt sich die Installation von RapidFuzz:
  ```bash
  pip install rapidfuzz
  ```

- **Streamlit React-Fehler (#185) oder Caching-Probleme:**
  Browser-Cache leeren / Seite neu laden oder Streamlit neu starten.
  Cache leeren:
  ```bash
  streamlit cache clear
  ```

---

## 📄 Lizenz

Dieses Projekt ist lizenziert unter der **MIT License**. Siehe [LICENSE](LICENSE) für Details.
