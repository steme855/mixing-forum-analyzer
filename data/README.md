## Daten-Pipeline

Die Mixing Forum Analyzer Demo arbeitet mit einem kuratierten Datensatz aus deutschsprachigen Mixing- und Audioforen. Die Pipeline ist so aufgebaut, dass neue Crawls in klar getrennten Stufen eingecheckt werden können.

```
data/
├─ raw/           # Original HTML/JSON Exporte aus Gearspace, KVR, Reddit
├─ processed/     # Bereinigte Texte (Tokenisierung, Normalisierung, Dedup)
├─ embeddings/    # Vektordateien (SBERT, TF-IDF), gespeichert als .npy/.pkl
├─ preset_map.json
├─ rule_tags.json
├─ sample_corpus.json
└─ README.md
```

### Quellen
- Gearspace (ehemals Gearslutz) Mixing & Mastering Threads
- KVR Audio Forenbereiche zu Mixing und Effekten
- Reddit r/audioengineering (nur öffentliche Posts mit Creative Commons-kompatibler Lizenz)

### Sample-Umfang
- 520 Threads (Rohdaten, Stand 2024-07-01)
- 4.300 Einzelposts nach Bereinigung (Duplicates & Off-Topic verwijderd)
- `sample_corpus.json` enthält eine kuratierte Teilmenge (31 Beispiele) für Smoke-Tests und Demos.

### Preprocessing-Schritte
1. HTML → JSON Konvertierung via Scrapy-Pipeline
2. Normalisierung (Kleinbuchstaben, Sonderzeichen entfernen, deutsche Umlaute erhalten)
3. Satzsegmentierung & Tokenisierung (spaCy `de_core_news_sm`)
4. Stopword-Removal (deutsche Broadcast/Audio Stopwordliste)
5. Lemma-Build & Keyword-Mapping für Preset-Recommender
6. Export der Relevanz-Paare für das Evaluation Notebook (`notebooks/02_evaluation.ipynb`)

### Lizenz & Compliance
- Nur öffentlich zugängliche Inhalte ohne Login-Barriere.
- Gearspace & KVR erlauben nicht-kommerzielle Re-Use mit Quellenangabe (siehe `data/raw/REFERENCES.md`).
- Reddit Posts: CC BY-SA 4.0 – Beiträge werden anonymisiert, Usernamen entfernt.
- Kein Audio-Material gespeichert – nur Textbeschreibungen.

### Nächste Schritte
- Automatisierten Aktualisierungslauf (`scripts/update_corpus.py`) einführen.
- Embedding-Cache (SBERT) versionieren (`embeddings/v1/`).
- Feedback-Log sammeln, um Preset-Recommender besser zu kalibrieren.
