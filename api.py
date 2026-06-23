"""FastAPI Entry Point – Mixing Forum Analyzer REST API."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# Stelle sicher, dass src/ auf dem Python-Pfad liegt
import sys
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api_access import APIKeyStore  # noqa: E402
from assistant_summary import ResponseSummarizer  # noqa: E402
from audio_analysis import analyze_wav_bytes  # noqa: E402
from feedback import FeedbackEntry, FeedbackStore  # noqa: E402
from personal_presets import PersonalPreset, PersonalPresetLibrary  # noqa: E402
from preset_advisor.search import SemanticSearchEngine  # noqa: E402
from presets.preset_recommender import PresetRecommender  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mixing Forum Analyzer API",
    description="AI Mixing Assistant mit Suche, Audioanalyse, Presets, Feedback und API-Key-Grundlage.",
    version="1.2.0",
)

# Engine einmalig beim Start instanziieren (corpus via data_ingestion)
_engine: SemanticSearchEngine | None = None
_feedback_store = FeedbackStore()
_summarizer = ResponseSummarizer()
_preset_recommender = PresetRecommender()
_personal_preset_library = PersonalPresetLibrary()
_api_key_store = APIKeyStore()


def _get_engine() -> SemanticSearchEngine:
    global _engine
    if _engine is None:
        logger.info("SemanticSearchEngine wird initialisiert …")
        _engine = SemanticSearchEngine()
    return _engine


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5
    use_sbert: bool = False
    include_summary: bool = False
    use_llm: bool = True


class SearchResultItem(BaseModel):
    doc_id: str
    text: str
    score: float


class AnalyzeResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResultItem]
    summary: Optional[dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    query: str
    rating: int
    comment: str = ""
    result_doc_ids: list[str] = Field(default_factory=list)
    feedback_type: str = "search"
    session_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    entry_id: str
    created_at: str
    status: str


class PersonalPresetRequest(BaseModel):
    user_id: str
    name: str
    plugin: str = ""
    category: str = "custom"
    description: str = ""
    settings: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class APIKeyIssueRequest(BaseModel):
    owner: str
    plan: str = "free"
    quota_limit: int = 100


@app.get("/health")
def health() -> dict[str, str]:
    """Einfacher Health-Check-Endpunkt."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: QueryRequest) -> AnalyzeResponse:
    """Durchsucht das Korpus und liefert optional eine Assistant-Zusammenfassung."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Leere Query ist nicht erlaubt.")

    engine = _get_engine()
    top_k = max(1, min(request.top_k, 20))
    results = engine.query_advanced(
        request.text,
        top_k=top_k,
        use_sbert=request.use_sbert,
    )
    mode = "SBERT" if request.use_sbert else "TF-IDF"

    summary = None
    if request.include_summary:
        presets = [
            f"{item.type}: {item.notes}"
            for item in _preset_recommender.suggest(request.text, top_k=5)
        ]
        summary = _summarizer.summarize(
            query=request.text,
            results=results,
            presets=presets,
            use_llm=request.use_llm,
        ).to_record()

    return AnalyzeResponse(
        query=request.text,
        mode=mode,
        results=[
            SearchResultItem(doc_id=r.doc_id, text=r.text, score=r.score)
            for r in results
        ],
        summary=summary,
    )


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def paid_analyze(request: QueryRequest, x_api_key: str = Header(default="")) -> AnalyzeResponse:
    """Quota-geschuetzter Analyze-Endpunkt als Monetarisierungsgrundlage."""
    try:
        _api_key_store.validate_and_consume(x_api_key, units=1, endpoint="/v1/analyze")
    except ValueError as exc:
        raise HTTPException(status_code=401 if "key" in str(exc).lower() else 402, detail=str(exc)) from exc
    return analyze(request)


@app.post("/audio/analyze")
async def analyze_audio(request: Request) -> dict[str, Any]:
    """Analysiert einen WAV-Upload aus dem Request-Body."""
    filename = request.headers.get("x-filename", "upload.wav")
    payload = await request.body()
    try:
        result = analyze_wav_bytes(payload, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result.to_record()


@app.post("/presets/personal")
def create_personal_preset(request: PersonalPresetRequest) -> dict[str, Any]:
    """Speichert ein persoenliches Preset fuer einen User."""
    try:
        preset = _personal_preset_library.add_preset(
            PersonalPreset(
                user_id=request.user_id,
                name=request.name,
                plugin=request.plugin,
                category=request.category,
                description=request.description,
                settings=request.settings,
                tags=tuple(request.tags),
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return preset.to_record()


@app.get("/presets/personal/{user_id}")
def list_personal_presets(user_id: str, category: Optional[str] = None) -> dict[str, Any]:
    """Listet persoenliche Presets fuer einen User."""
    presets = _personal_preset_library.list_presets(user_id, category=category)
    return {"user_id": user_id, "presets": [preset.to_record() for preset in presets]}


@app.delete("/presets/personal/{user_id}/{preset_id}")
def delete_personal_preset(user_id: str, preset_id: str) -> dict[str, Any]:
    """Entfernt ein persoenliches Preset."""
    deleted = _personal_preset_library.delete_preset(user_id, preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"status": "deleted", "preset_id": preset_id}


@app.post("/api-keys")
def issue_api_key(request: APIKeyIssueRequest) -> dict[str, Any]:
    """Erstellt einen API-Key. Der Klartext-Key wird nur einmal zurueckgegeben."""
    try:
        issued = _api_key_store.issue_key(
            owner=request.owner,
            plan=request.plan,
            quota_limit=request.quota_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "key_id": issued.key_id,
        "api_key": issued.api_key,
        "owner": issued.owner,
        "plan": issued.plan,
        "quota_limit": issued.quota_limit,
    }


@app.get("/api-keys/{key_id}")
def api_key_status(key_id: str) -> dict[str, Any]:
    """Zeigt Key-Status ohne Klartext-Key."""
    record = _api_key_store.get_record(key_id)
    if record is None:
        raise HTTPException(status_code=404, detail="API key not found")
    data = record.to_record()
    data.pop("key_hash", None)
    return data


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Speichert Nutzerfeedback für Ranking-, Preset- und Summary-Qualität."""
    try:
        entry = _feedback_store.append(
            FeedbackEntry(
                query=request.query,
                rating=request.rating,
                comment=request.comment,
                result_doc_ids=tuple(request.result_doc_ids),
                feedback_type=request.feedback_type,
                session_id=request.session_id,
                metadata=request.metadata,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return FeedbackResponse(entry_id=entry.entry_id, created_at=entry.created_at, status="stored")


@app.get("/feedback/summary")
def feedback_summary() -> dict[str, Any]:
    """Liefert einfache Feedback-Kennzahlen für Qualitätsmonitoring."""
    return _feedback_store.summary()
