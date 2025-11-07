"""Heuristics for mapping textual mix issues to actionable preset ideas."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from preset_advisor.core import normalize_text


SEVERITY_FACTORS = {
    "light": 0.5,
    "leicht": 0.5,
    "mild": 0.5,
    "medium": 1.0,
    "mittel": 1.0,
    "moderate": 1.0,
    "strong": 1.5,
    "stark": 1.5,
    "heavy": 1.5,
}


@dataclass(frozen=True)
class PresetRecommendation:
    label: str
    type: str
    frequency: int | None
    shelf: str | None
    gain_db: float | None
    q: float | None
    severity: str
    notes: str
    score: float


SYMPTOM_LIBRARY = [
    {
        "label": "blechern",
        "keywords": ["blechern", "metallisch", "3khz", "zu viel 3khz", "zu viel 3 khz"],
        "type": "EQ Bell Cut",
        "frequency": 3000,
        "q": 2.2,
        "gain_db": -3.0,
        "notes": "Schmalbandiger Cut bei 3-5 kHz reduziert Blechanteile.",
    },
    {
        "label": "kick_gain_control",
        "keywords": [
            "kick zu laut",
            "kick ist zu laut",
            "kick laut",
            "kick pegel",
            "kick volume",
        ],
        "type": "Fader Trim",
        "frequency": 100,
        "q": None,
        "gain_db": -2.5,
        "notes": "Kick-Bus um 2-3 dB absenken und Low-End neu balancieren.",
    },
    {
        "label": "sibilance",
        "keywords": ["s-laute", "sibilance", "zischend", "sch", "zisch"],
        "type": "De-Esser",
        "frequency": 7000,
        "q": None,
        "gain_db": -4.0,
        "notes": "Schmalbandiger De-Esser im Bereich 6-8 kHz.",
    },
    {
        "label": "muddy",
        "keywords": ["mumpfig", "muddy", "zu viel 200hz", "200hz", "low mids"],
        "type": "EQ Bell Cut",
        "frequency": 250,
        "q": 1.4,
        "gain_db": -3.5,
        "notes": "Low-Mid Cut schafft Platz für Kick und Bass.",
    },
    {
        "label": "lacks_punch",
        "keywords": ["kein punch", "zu weich", "kick zu weich", "attack fehlt"],
        "type": "Transient Designer",
        "frequency": None,
        "q": None,
        "gain_db": 4.0,
        "notes": "Attack erhöhen, Sustain leicht absenken.",
    },
    {
        "label": "harsh_highs",
        "keywords": ["harsch", "scharf", "zu viel 8khz", "zu viel höhen", "zu viel 8 khz"],
        "type": "EQ Shelf Cut",
        "frequency": 8000,
        "q": None,
        "gain_db": -2.5,
        "notes": "High-Shelf sanft absenken für smootheres Top-End.",
    },
    {
        "label": "needs_presence",
        "keywords": ["zu dumpf", "mehr präsenz", "fehlt präsenz", "3khz push"],
        "type": "EQ Bell Boost",
        "frequency": 3200,
        "q": 1.0,
        "gain_db": 2.5,
        "notes": "Leichter Boost bei 3 kHz bringt Stimmen nach vorne.",
    },
]


def _tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return re.split(r"[\s,;.!?]+", normalized)


class PresetRecommender:
    """Keyword-driven preset recommender with severity-aware gain suggestions."""

    def __init__(self, library: Iterable[dict] | None = None) -> None:
        self._rules = list(library or SYMPTOM_LIBRARY)

    def suggest(self, problem: str, severity: str = "medium", top_k: int | None = None) -> list[PresetRecommendation]:
        tokens = set(_tokenize(problem))
        severity_key = normalize_text(severity)
        factor = SEVERITY_FACTORS.get(severity_key, 1.0)
        matches: list[PresetRecommendation] = []

        for rule in self._rules:
            keywords = {normalize_text(keyword) for keyword in rule["keywords"]}
            if tokens.isdisjoint(keywords) and not any(keyword in normalize_text(problem) for keyword in keywords):
                continue

            gain_db = rule.get("gain_db")
            adjusted_gain = float(gain_db * factor) if gain_db is not None else None
            score = 1.0 * factor

            matches.append(
                PresetRecommendation(
                    label=rule.get("label", "generic"),
                    type=rule.get("type", "Preset"),
                    frequency=rule.get("frequency"),
                    shelf=rule.get("shelf"),
                    gain_db=adjusted_gain,
                    q=rule.get("q"),
                    severity=severity_key or "medium",
                    notes=rule.get("notes", ""),
                    score=score,
                )
            )

        matches.sort(key=lambda rec: rec.score, reverse=True)
        if top_k is not None:
            return matches[:top_k]
        return matches
