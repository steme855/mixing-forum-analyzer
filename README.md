
# Mixing Forum Analyzer

**Kurz:** Eine kleine NLP-Demo, die Mixing-Posts (Kick, Snare, Vocals etc.) analysiert:
- Tokenisierung → TF‑IDF → Ähnlichkeit (Cosine)
- Streamlit‑App: Texteingabe und „ähnliche Posts“ finden
- spaCy‑Analyse: Wortarten, Lemmata, Sound‑Adjektive

## Setup (Mac & Linux)
```bash
git clone <YOUR_REPO_URL> mixing-forum-analyzer
cd mixing-forum-analyzer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
```

## Start (lokal)
```bash
streamlit run app/app.py
```

## Start (Linux‑Server, Port 8080)
```bash
source venv/bin/activate
streamlit run app/app.py --server.port 8080 --server.headless true
# dann: http://SERVER_IP:8080
```

## Verzeichnisstruktur
```
mixing-forum-analyzer/
├─ app/                 # Streamlit-App
├─ data/                # Beispiel-Daten
├─ notebooks/           # Jupyter-Notebooks (Lernen/Eval)
├─ scripts/             # Hilfsskripte (Setup, Run)
├─ docker/              # Docker-Setup (optional)
└─ tests/               # Smoke-Tests
```

## Roadmap Woche 1
- `01_tokenizer.ipynb`: Tokenisierung (Regex, CountVectorizer)
- `02_tfidf.ipynb`: TF‑IDF + Top-Wörter
- `03_similarity.ipynb`: Cosine Similarity
- `04_spacy.ipynb`: POS & Lemmata; Sound‑Adjektive extrahieren
- `app.py`: Streamlit‑Demo (TF‑IDF + Similarity)

## Lizenz
MIT
