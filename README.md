<!-- markdownlint-disable MD013 -->

# Mixing Forum Analyzer

Mixing Forum Analyzer (MFA) is a local-first prototype for searching, analyzing, and summarizing recurring audio mixing problems. It combines a curated demo knowledge base, TF-IDF/SBERT retrieval, rule-based preset guidance, assistant-style summaries, basic WAV analysis, feedback capture, personal preset storage, and API access primitives.

[![CI/CD](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_and_deploy.yml)
[![Tests](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_only.yml/badge.svg)](https://github.com/steme855/mixing-forum-analyzer/actions/workflows/test_only.yml)
[![Codecov](https://codecov.io/gh/steme855/mixing-forum-analyzer/branch/main/graph/badge.svg)](https://codecov.io/gh/steme855/mixing-forum-analyzer)
[![License: MIT](https://img.shields.io/github/license/steme855/mixing-forum-analyzer.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)](pyproject.toml)

MFA is useful for:

- audio engineers who want to find similar mix-problem cases faster
- developers building retrieval and assistant workflows for music-production domains
- educators and creators testing local knowledge-base workflows for mixing advice
- SaaS/API evaluators assessing whether the project can become a productized mixing assistant

The repository is a prototype. It does not claim production-grade retrieval quality, production billing, licensed large-scale forum coverage, or finished DAW/VST integration.

## Demo

![Mixing Forum Analyzer demo](docs/demo.gif)

## Current Functionality

| Area | Current implementation |
| ---- | ---------------------- |
| Search | TF-IDF search with optional SBERT embeddings via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. |
| Corpus ingestion | `src/data_ingestion/` normalizes JSON, JSONL/NDJSON, and CSV forum exports into a deduplicated knowledge base. |
| Preset coach | `presets/preset_recommender.py` maps textual symptoms to EQ, dynamics, and gain suggestions using deterministic rules. |
| Assistant summaries | `src/assistant_summary/` creates concise answers from search evidence, with optional OpenAI API usage and a deterministic local fallback. |
| Feedback | `src/feedback/` stores search, preset, and summary ratings in append-only JSONL files. |
| WAV analysis | `src/audio_analysis/` analyzes PCM WAV uploads for peak level, RMS, crest factor, clipping samples, silence ratio, and dominant frequency. |
| Personal presets | `src/personal_presets/` stores user-owned preset libraries as local JSON files. |
| API access foundation | `src/api_access/` issues hashed API keys and tracks quota usage for protected API routes. |
| Evaluation | `evaluation/metrics.py` calculates MRR, top-k accuracy, latency summaries, and similarity distributions. |

## Quickstart

```bash
git clone https://github.com/steme855/mixing-forum-analyzer.git
cd mixing-forum-analyzer

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download de_core_news_sm

streamlit run app.py
```

On Windows PowerShell:

```powershell
git clone https://github.com/steme855/mixing-forum-analyzer.git
cd mixing-forum-analyzer

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download de_core_news_sm

streamlit run app.py
```

The current demo still uses a German spaCy model because parts of the original prototype and sample corpus include German-language mixing phrases.

## API Usage

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

Run a basic search:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"The kick drum is too loud in the mix","top_k":5,"use_sbert":false}'
```

Request a search with an assistant summary:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"The vocal sibilance sounds too harsh","top_k":5,"include_summary":true,"use_llm":false}'
```

Analyze a WAV file:

```bash
curl -X POST http://127.0.0.1:8000/audio/analyze \
  -H "Content-Type: audio/wav" \
  -H "x-filename: mix.wav" \
  --data-binary @mix.wav
```

Create an API key and call the quota-protected endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api-keys \
  -H "Content-Type: application/json" \
  -d '{"owner":"local-test","plan":"free","quota_limit":100}'

curl -X POST http://127.0.0.1:8000/v1/analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: <returned-api-key>" \
  -d '{"text":"The bass is masking the kick drum","top_k":5}'
```

## Development Commands

```bash
make install       # install runtime dependencies
make dev-install   # install runtime and development dependencies
make lint          # flake8, black --check, isort --check-only
make format        # black and isort
make test          # pytest
make coverage      # pytest with coverage output
```

If optional pytest plugins are not installed in your local environment, run:

```bash
python -m pytest -o addopts='' tests -q
```

## Architecture

```text
mixing-forum-analyzer/
├── app.py                         # Streamlit launcher
├── api.py                         # FastAPI entry point
├── data/
│   ├── sample_corpus.json         # small curated demo corpus
│   ├── processed/                 # processed exports and evaluation outputs
│   ├── feedback/                  # runtime feedback JSONL, ignored by git
│   ├── personal_presets/          # runtime user preset JSON, ignored by git
│   └── api_keys/                  # runtime key and usage data, ignored by git
├── evaluation/
│   └── metrics.py                 # ranking and latency evaluation helpers
├── presets/
│   └── preset_recommender.py      # rule-based preset suggestions
├── src/
│   ├── app/main.py                # Streamlit UI
│   ├── api_access/                # API keys and quota accounting
│   ├── assistant_summary/         # summary wrapper with local fallback
│   ├── audio_analysis/            # PCM WAV metrics
│   ├── data_ingestion/            # corpus normalization and deduplication
│   ├── feedback/                  # feedback storage
│   ├── personal_presets/          # user preset libraries
│   └── preset_advisor/            # search and text utilities
└── tests/                         # unit and integration tests
```

### Data Flow

1. Forum data is loaded from `data/sample_corpus.json`, `data/processed/`, and future `data/raw/` exports.
2. `src/data_ingestion/` normalizes records, attaches available metadata, and deduplicates text entries.
3. `src/preset_advisor/search.py` indexes the corpus with TF-IDF and optional SBERT.
4. Streamlit and FastAPI call the same search, summary, feedback, audio-analysis, and preset modules.
5. Feedback, user presets, API-key usage, and audio-analysis outputs are stored locally for MVP workflows.

## Search Backends

| Aspect | TF-IDF | SBERT |
| ------ | ------ | ----- |
| Dependency profile | Lightweight and deterministic | Requires loading a `sentence-transformers` model |
| Best use case | Short keyword-style problems and offline mode | More flexible natural-language phrasing |
| Index input | Ingested forum texts | Ingested forum texts |
| Scoring | Cosine similarity in TF-IDF space | Cosine similarity over normalized embeddings |

SBERT is optional at runtime. If the embedding model cannot be loaded, MFA can still operate with TF-IDF.

## Evaluation

The current benchmark is based on 29 queries from `notebooks/02_evaluation.ipynb`. The reported ranking metrics are useful for regression checks on the small curated benchmark, but they should not be interpreted as proof of production-level retrieval quality.

| Metric | Current value | Benchmark target |
| ------ | ------------: | ---------------- |
| MRR | 1.00 | > 0.70 |
| Top-1 Accuracy | 1.00 | > 0.60 |
| Top-3 Accuracy | 1.00 | > 0.80 |
| Top-5 Accuracy | 1.00 | > 0.90 |

Source: `docs/metrics_for_readme.md` and `notebooks/02_evaluation.ipynb`.

Recommended next evaluation steps:

- expand the query set beyond the current 29-query benchmark
- add held-out forum posts and failure cases
- compare TF-IDF and SBERT under the same labeled relevance set
- track latency and quality separately for local, API, and deployment environments

## Business Value

MFA targets a practical workflow problem: engineers often search old forum threads for recurring issues such as harsh vocals, kick/bass masking, boxy drums, dull vocals, or over-compressed mix buses. The project can reduce manual browsing when the knowledge base contains similar cases and when the retrieved suggestions fit the session context.

The business case is strongest where MFA becomes:

- a searchable knowledge base for repeated mix problems
- a feedback-driven assistant that improves recommendations over time
- an API layer for audio tools, education products, or studio workflows
- a bridge between text-based mix diagnosis and preset or DAW automation

This README intentionally avoids unverified time-savings, revenue claims, and production-readiness claims.

## Data and Compliance Notes

See `data/README.md` for the current data structure, source notes, and preprocessing assumptions.

Important limitations:

- the repository currently includes a small curated sample corpus and seed data
- future large-scale forum ingestion needs source-specific license and terms review
- runtime feedback, API-key data, and personal presets can contain user-specific data and are ignored by git
- WAV analysis currently supports PCM WAV metrics, not stem separation or perceptual mix critique
- optional LLM summaries should be evaluated for grounding, safety, and source attribution before production use

## Roadmap

The following items are roadmap or production-hardening work unless explicitly listed above as current implementation.

| Area | Roadmap direction |
| ---- | ----------------- |
| Larger forum dataset | Move from sample and processed demo data to a larger, licensed, source-tracked knowledge base. |
| User feedback system | Use collected ratings to improve ranking, preset suggestions, and summary quality. |
| LLM answer summaries | Improve prompt design, source grounding, evaluation, and safety controls for generated mixing advice. |
| WAV upload audio analysis | Extend basic PCM metrics toward richer audio diagnostics and clearer UI interpretation. |
| Personal preset libraries | Add account-backed storage, import/export, tagging, and retrieval against user context. |
| API monetization | Replace local JSON key storage with production authentication, billing, plan limits, abuse controls, and analytics. |
| DAW/VST integration | Define plugin-facing schemas and sync workflows for DAW sessions, presets, and assistant recommendations. |

## Project Links

- Setup and deployment notes: `docs/setup_guide.md`
- Evaluation helper output: `docs/metrics_for_readme.md`
- Data pipeline notes: `data/README.md`
- License: `LICENSE`

## Author

Stephan Meyer - Audio Engineer and Developer
