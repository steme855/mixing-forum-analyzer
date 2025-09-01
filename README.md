# 🎚 Mixing Forum Analyzer  

> *KI-Demo: Automatische Analyse und Suche nach ähnlichen Mixing-Problemen in Foren-Posts (Kick, Snare, Vocals etc.)*  
> *Ziel: KI-gestützte Mixing-Support-Tools für Engineers, Education und Plugin-Hersteller.*  

[![CI](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/steme855/mixing-forum-analyzer/actions)  
[![Open in Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Open%20Space-blue)](https://huggingface.co/spaces/Stepman/mixing-forum-analyzer)

---

## 🚀 Live-Demo

👉 Direkt ausprobieren im Hugging Face Space:  
[https://huggingface.co/spaces/Stepman/mixing-forum-analyzer](https://huggingface.co/spaces/Stepman/mixing-forum-analyzer)  

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
pip install -r requirements.txt
streamlit run app/app.py
