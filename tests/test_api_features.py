from __future__ import annotations

import io
import math
import wave

from fastapi.testclient import TestClient

import api
from api_access import APIKeyStore
from feedback import FeedbackStore
from personal_presets import PersonalPresetLibrary


def _sine_wav_bytes(frequency: float = 440.0, sample_rate: int = 8000, seconds: float = 0.1) -> bytes:
    frames = int(sample_rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        samples = bytearray()
        for index in range(frames):
            value = int(0.5 * 32767 * math.sin(2 * math.pi * frequency * index / sample_rate))
            samples.extend(value.to_bytes(2, byteorder="little", signed=True))
        wav.writeframes(bytes(samples))
    return buffer.getvalue()


def test_analyze_can_include_fallback_summary() -> None:
    client = TestClient(api.app)

    response = client.post(
        "/analyze",
        json={
            "text": "Kick zu laut",
            "top_k": 3,
            "use_sbert": False,
            "include_summary": True,
            "use_llm": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    assert payload["summary"]["mode"] == "fallback"
    assert payload["summary"]["action_steps"]


def test_feedback_endpoint_stores_entry(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(tmp_path / "feedback.jsonl"))
    client = TestClient(api.app)

    response = client.post(
        "/feedback",
        json={
            "query": "Snare harsch",
            "rating": 4,
            "comment": "brauchbar",
            "result_doc_ids": ["doc_1"],
            "feedback_type": "summary",
        },
    )
    summary_response = client.get("/feedback/summary")

    assert response.status_code == 200
    assert response.json()["status"] == "stored"
    assert summary_response.status_code == 200
    assert summary_response.json()["count"] == 1
    assert summary_response.json()["by_type"] == {"summary": 1}


def test_feedback_endpoint_rejects_invalid_rating(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "_feedback_store", FeedbackStore(tmp_path / "feedback.jsonl"))
    client = TestClient(api.app)

    response = client.post(
        "/feedback",
        json={"query": "Snare harsch", "rating": 0},
    )

    assert response.status_code == 422


def test_audio_analyze_endpoint_accepts_wav_body() -> None:
    client = TestClient(api.app)

    response = client.post(
        "/audio/analyze",
        content=_sine_wav_bytes(),
        headers={"x-filename": "sine.wav", "content-type": "audio/wav"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sample_rate"] == 8000
    assert payload["dominant_frequency_hz"] is not None


def test_personal_preset_endpoints_store_and_list(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "_personal_preset_library", PersonalPresetLibrary(tmp_path))
    client = TestClient(api.app)

    create_response = client.post(
        "/presets/personal",
        json={
            "user_id": "user-1",
            "name": "Kick Cleanup",
            "plugin": "Pro-Q",
            "category": "eq",
            "tags": ["kick"],
        },
    )
    list_response = client.get("/presets/personal/user-1")

    assert create_response.status_code == 200
    assert list_response.status_code == 200
    assert list_response.json()["presets"][0]["name"] == "Kick Cleanup"


def test_api_key_flow_protects_v1_analyze(tmp_path, monkeypatch) -> None:
    store = APIKeyStore(tmp_path)
    monkeypatch.setattr(api, "_api_key_store", store)
    client = TestClient(api.app)

    issue_response = client.post(
        "/api-keys",
        json={"owner": "studio-a", "plan": "free", "quota_limit": 1},
    )
    issued = issue_response.json()
    analyze_response = client.post(
        "/v1/analyze",
        json={"text": "Kick zu laut", "top_k": 2},
        headers={"x-api-key": issued["api_key"]},
    )
    quota_response = client.post(
        "/v1/analyze",
        json={"text": "Kick zu laut", "top_k": 2},
        headers={"x-api-key": issued["api_key"]},
    )
    status_response = client.get(f"/api-keys/{issued['key_id']}")

    assert issue_response.status_code == 200
    assert analyze_response.status_code == 200
    assert quota_response.status_code == 402
    assert status_response.json()["quota_used"] == 1
    assert "key_hash" not in status_response.json()
