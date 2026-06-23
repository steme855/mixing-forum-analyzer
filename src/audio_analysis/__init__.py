"""WAV analysis utilities for the AI Mixing Assistant."""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AudioAnalysisResult:
    """Compact technical summary of an uploaded WAV file."""

    sample_rate: int
    channels: int
    duration_seconds: float
    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    clipping_samples: int
    silence_ratio: float
    dominant_frequency_hz: float | None
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "peak_dbfs": self.peak_dbfs,
            "rms_dbfs": self.rms_dbfs,
            "crest_factor_db": self.crest_factor_db,
            "clipping_samples": self.clipping_samples,
            "silence_ratio": self.silence_ratio,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "notes": list(self.notes),
            "metadata": self.metadata,
        }


def analyze_wav_bytes(payload: bytes, filename: str = "upload.wav") -> AudioAnalysisResult:
    """Analyze PCM WAV bytes using only stdlib + numpy."""

    if not payload:
        raise ValueError("WAV payload is empty")

    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            raw = wav.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError(f"Invalid WAV payload: {exc}") from exc

    if channels < 1 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV file has no analyzable audio frames")

    samples = _decode_pcm(raw, sample_width)
    if samples.size == 0:
        raise ValueError("WAV file has no PCM samples")

    sample_matrix = samples.reshape(-1, channels) if channels > 1 else samples.reshape(-1, 1)
    mono = sample_matrix.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono))))
    peak_dbfs = _amp_to_dbfs(peak)
    rms_dbfs = _amp_to_dbfs(rms)
    crest = peak_dbfs - rms_dbfs if math.isfinite(peak_dbfs) and math.isfinite(rms_dbfs) else 0.0
    clipping_samples = int(np.sum(np.max(np.abs(sample_matrix), axis=1) >= 0.999))
    silence_ratio = float(np.mean(np.abs(mono) < 10 ** (-60 / 20)))
    dominant_frequency = _dominant_frequency(mono, sample_rate)

    return AudioAnalysisResult(
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=round(frame_count / sample_rate, 3),
        peak_dbfs=round(peak_dbfs, 2),
        rms_dbfs=round(rms_dbfs, 2),
        crest_factor_db=round(crest, 2),
        clipping_samples=clipping_samples,
        silence_ratio=round(silence_ratio, 4),
        dominant_frequency_hz=round(dominant_frequency, 1) if dominant_frequency is not None else None,
        notes=_build_notes(peak_dbfs, rms_dbfs, crest, clipping_samples, silence_ratio),
        metadata={"filename": filename, "sample_width_bytes": sample_width, "frames": frame_count},
    )


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return data / 32768.0
    if sample_width == 3:
        raw_array = np.frombuffer(raw, dtype=np.uint8)
        if raw_array.size % 3 != 0:
            raise ValueError("Invalid 24-bit PCM byte count")
        triplets = raw_array.reshape(-1, 3).astype(np.int32)
        values = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        return data / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def _amp_to_dbfs(value: float) -> float:
    if value <= 0:
        return -120.0
    return max(-120.0, 20 * math.log10(value))


def _dominant_frequency(mono: np.ndarray, sample_rate: int) -> float | None:
    if mono.size < 16:
        return None
    window_size = min(mono.size, sample_rate * 2)
    segment = mono[:window_size]
    segment = segment - np.mean(segment)
    if np.max(np.abs(segment)) < 1e-6:
        return None
    windowed = segment * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.size, d=1 / sample_rate)
    if spectrum.size <= 1:
        return None
    spectrum[0] = 0
    index = int(np.argmax(spectrum))
    return float(freqs[index])


def _build_notes(
    peak_dbfs: float,
    rms_dbfs: float,
    crest_factor_db: float,
    clipping_samples: int,
    silence_ratio: float,
) -> tuple[str, ...]:
    notes: list[str] = []
    if clipping_samples:
        notes.append("Clipping erkannt: Peak- oder Limiter-Stufe prüfen.")
    if peak_dbfs > -1.0:
        notes.append("Sehr hoher Peak-Level: Headroom vor weiterer Bearbeitung schaffen.")
    if rms_dbfs > -10.0:
        notes.append("Hoher RMS-Level: Mix kann bereits stark verdichtet sein.")
    if crest_factor_db < 6.0:
        notes.append("Niedriger Crest-Faktor: Transienten und Bus-Kompression prüfen.")
    if silence_ratio > 0.5:
        notes.append("Hoher Stilleanteil: Datei auf Beschnitt oder Exportbereich prüfen.")
    if not notes:
        notes.append("Keine offensichtlichen technischen Warnsignale im WAV-Upload.")
    return tuple(notes)
