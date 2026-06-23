from __future__ import annotations

import io
import math
import wave

import pytest

from audio_analysis import analyze_wav_bytes


def _sine_wav_bytes(frequency: float = 440.0, sample_rate: int = 8000, seconds: float = 0.25) -> bytes:
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


def test_analyze_wav_bytes_returns_core_metrics() -> None:
    result = analyze_wav_bytes(_sine_wav_bytes(), filename="sine.wav")

    assert result.sample_rate == 8000
    assert result.channels == 1
    assert result.duration_seconds == 0.25
    assert -7.0 < result.peak_dbfs < -5.0
    assert result.dominant_frequency_hz is not None
    assert abs(result.dominant_frequency_hz - 440.0) < 20.0


def test_analyze_wav_bytes_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="Invalid WAV"):
        analyze_wav_bytes(b"not a wav")
