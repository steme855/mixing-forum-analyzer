# 🎚 Mixing Forum Analyzer  

> *KI-Demo: Automatische Analyse und Suche nach ähnlichen Mixing-Problemen in Foren-Posts (Kick, Snare, Vocals etc.)*  
> *Ziel: KI-gestützte Mixing-Support-Tools für Engineers, Education und Plugin-Hersteller.*

---

## 📸 Demo

👉 Beispiel: User tippt *„Snare zu laut“* → Tool findet ähnliche Posts.  
*(Hier dein GIF einfügen, z. B. mit [LICEcap](https://www.cockos.com/licecap/)):*

![Demo GIF](demo.gif)

---

## ⚡ Features

- **Tokenisierung → TF-IDF → Cosine Similarity**  
- **Streamlit-App**: Texteingabe & „ähnliche Posts“ finden  
- **spaCy-Analyse**: Wortarten, Lemmata, Sound-Adjektive  
- **Preset-Map**: Erste Mapping-Ideen basierend auf *Handbuch der Tonstudiotechnik*  

---

## 🚀 Setup (Mac & Linux)

```bash
git clone https://github.com/<YOUR_USERNAME>/mixing-forum-analyzer.git
cd mixing-forum-analyzer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
