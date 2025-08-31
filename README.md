![CI](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/ci.yml/badge.svg)
# 🎚 Mixing Forum Analyzer  

> *KI-Demo: Automatische Analyse und Suche nach ähnlichen Mixing-Problemen in Foren-Posts (Kick, Snare, Vocals etc.)*  
> *Ziel: KI-gestützte Mixing-Support-Tools für Engineers, Education und Plugin-Hersteller.*

---

## 📸 Demo

👉 Beispiel: User tippt *„Snare zu laut“* → Tool findet ähnliche Posts.  
*(Hier dein GIF einfügen, z. B. mit [LICEcap](https://www.cockos.com/licecap/)):*

![Demo GIF](app/assets/demo.gif)

## 🚀 Live-Demo (Hugging Face Space)

[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open%20Space-blue)](https://huggingface.co/spaces/Stepman/mixing-forum-analyzer)

Die Space startet die Streamlit-App. Beim ersten semantischen Suchlauf wird das SBERT-Modell (~80 MB) automatisch geladen/cached.

---

## ⚡ Features

- **Tokenisierung → TF-IDF → Cosine Similarity**  
- **Streamlit-App**: Texteingabe & „ähnliche Posts“ finden  
- **spaCy-Analyse**: Wortarten, Lemmata, Sound-Adjektive  
- **Preset-Map**: Erste Mapping-Ideen basierend auf *Handbuch der Tonstudiotechnik*  

---

## 🚀 Setup (Mac & Linux)

```bash
git clone https://github.com/steme855/mixing-forum-analyzer.git
cd mixing-forum-analyzer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm


**Leise Logs:** Die App unterdrückt PyTorch/Torch-C++-Init-Logs für ein ruhigeres Terminal.

**spaCy DE:** Wenn die rechte Analyse-Spalte leer bleibt, installiere das deutsche Modell:
```bash
python -m spacy download de_core_news_sm
