---
title: "Mixing Forum Analyzer"
emoji: "🎚"
colorFrom: "red"
colorTo: "purple"
sdk: "streamlit"
sdk_version: "1.38.0"
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

---

## 🚀 Live-Demo

👉 Direkt ausprobieren im Hugging Face Space:  
https://huggingface.co/spaces/Stepman/mixing-forum-analyzer

![Demo GIF](app/assets/demo.gif)

---

## ⚡ Features

- **Semantische Suche (SBERT)** – ähnliche Mixing-Probleme in Foren finden  
- **TF-IDF Fallback** – falls SBERT auf dem Host nicht verfügbar ist  
- **spaCy-Analyse (DE)** – Wortarten, Lemmata, „Lemma-Overlap“-Erklärungen  
- **Preset-Map** – funktioniert mit *flacher Liste* **oder** *strukturiertem Schema*  
- **CSV/JSON Export** – Ergebnisse & Preset-Vorschläge exportieren  
- **UI Goodies** – Demo-GIF, Enter-to-send, Mini-Bar-Chart, Status-Badges

---

## 🛠 Setup (lokal)

```bash
git clone https://github.com/steme855/mixing-forum-analyzer
cd mixing-forum-analyzer

python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# Optional (nur lokal, NICHT auf Hugging Face nötig):
python -m spacy download de_core_news_sm
```
# Für schnelleres Hot-Reload (empfohlen):
python -m pip install watchdog

streamlit run app/app.py
