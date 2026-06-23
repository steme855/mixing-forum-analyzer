# Daten-Pipeline

Die Mixing Forum Analyzer Demo arbeitet aktuell mit einem kleinen kuratierten
Seed-Datensatz aus deutschsprachigen Mixing-Problemsätzen.

Die Pipeline ist so aufgebaut, dass spätere, lizenzgeprüfte Forum-Crawls in
klar getrennten Stufen ergänzt werden können.

```text
data/
├─ raw/           # Original HTML/JSON Exporte aus Gearspace, KVR, Reddit
├─ processed/     # Bereinigte Texte (Tokenisierung, Normalisierung, Dedup)
├─ embeddings/    # Vektordateien (SBERT, TF-IDF), gespeichert als .npy/.pkl
├─ feedback/      # Nutzerfeedback als JSONL für Qualitätsauswertung
├─ personal_presets/ # Nutzerbezogene Preset-Bibliotheken
├─ api_keys/      # Gehashte API-Key- und Usage-Daten
├─ dataset_manifest.json
├─ preset_map.json
├─ rule_tags.json
├─ sample_corpus.json
└─ README.md
```

## Quellen

- Aktuell versioniert: kuratierte/synthetische Seed-Beispiele.
- Nicht enthalten: vollständige Gearspace-, KVR- oder Reddit-Crawls.
- Zukünftige Forum-Crawls benötigen Lizenz- und Datenschutzprüfung.

## Sample-Umfang

- `sample_corpus.json`: 32 kuratierte Demo-Dokumente.
- `raw/seed_forum_posts.jsonl`: Seed-Dokumente mit Metadaten.
- `processed/corpus.csv`: normalisierte CSV-Ansicht des Korpus.
- `processed/relevance_pairs.csv`: 29 Query-zu-Dokument-Paare.
- `dataset_manifest.json`: maschinenlesbare Datensatzbeschreibung.

## Preprocessing-Schritte

1. Seed-Dokumente werden als JSON und JSONL abgelegt.
2. `processed/corpus.csv` hält eine normalisierte tabellarische Sicht.
3. `processed/keyphrases.csv` ergänzt Keyphrases pro Seed-Dokument.
4. `processed/relevance_pairs.csv` definiert Evaluation-Zuordnungen.
5. `src/data_ingestion/` lädt JSON, JSONL/NDJSON und CSV.
6. Die Ingestion dedupliziert anhand normalisierter Text-Fingerprints.

## Lizenz & Compliance

- Versionierte Seed-Daten sind als `synthetic_or_curated_example` markiert.
- Keine echten Nutzerprofile, Usernamen oder Audio-Dateien sind Teil des Seed-Datensatzes.
- Große Forum-Crawls sind Roadmap-Arbeit.
- Externe Daten dürfen erst nach Lizenz-/Datenschutzprüfung ergänzt werden.
- Runtime-Daten sind nicht Teil des statischen Datensatzes.

## Nächste Schritte

- Automatisierten Aktualisierungslauf (`scripts/update_corpus.py`) einführen.
- Embedding-Cache (SBERT) versionieren (`embeddings/v1/`).
- Größere Forum-Crawls nur mit Quellenmanifest ergänzen.
- Feedback-Log sammeln, um Preset-Recommender besser zu kalibrieren.
- KVR-Pilot-Crawler zunächst im Dry-Run prüfen.

## Neue Ingestion-Schicht

- `src/data_ingestion/` lädt Forumdaten aus JSON, JSONL und CSV.
- Unterstützte Formate: JSON, JSONL/NDJSON und CSV.
- Deduplizierung läuft über normalisierte Text-Fingerprints.
- `processed/keyphrases.csv` wird als Metadaten-Index genutzt.
- Search-Engine und Streamlit-App konsumieren dieselbe Ingestion-Schicht.

## KVR-Pilot-Crawler

`scripts/crawl_kvr_pilot.py` ist ein compliance-first Pilot-Crawler.

Standardmodus:

```bash
python scripts/crawl_kvr_pilot.py \
  --url "https://www.kvraudio.com/forum/viewtopic.php?t=THREAD_ID"
```

Das schreibt nur ein Dry-Run-Manifest und crawlt keine Forumseiten.

Echte Abrufe erfordern bewusst `--execute`:

```bash
python scripts/crawl_kvr_pilot.py \
  --execute \
  --max-threads 10 \
  --delay-seconds 3 \
  --url-file data/raw/kvr_seed_urls.txt
```

Runtime-Ausgaben wie `kvr_pilot_manifest.json` und `kvr_pilot_crawl.jsonl`
werden nicht versioniert.

## Feedback & Summary MVP

- `src/feedback/` speichert Nutzerbewertungen append-only in `data/feedback/feedback.jsonl`.
- Feedback-Typen sind aktuell `search`, `preset` und `summary`.
- `/feedback/summary` liefert einfache Qualitätskennzahlen.
- `src/assistant_summary/` erzeugt Antwortzusammenfassungen.
- Ohne `OPENAI_API_KEY` arbeitet die Summary lokal regelbasiert.

## Audio, Presets & API-Monetarisierung

- `src/audio_analysis/` analysiert PCM-WAV-Dateien.
- Metriken: Peak, RMS, Crest-Faktor, Clipping und dominante Frequenz.
- `src/personal_presets/` speichert persönliche Preset-Bibliotheken.
- `src/api_access/` erstellt gehashte API-Keys und zählt Quota-Verbrauch.
