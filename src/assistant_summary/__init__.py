"""LLM-backed response summaries with deterministic fallback behavior."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from preset_advisor.search import SearchResult


@dataclass(frozen=True)
class MixingSummary:
    """Structured assistant response returned by API and UI."""

    query: str
    answer: str
    action_steps: tuple[str, ...]
    caveats: tuple[str, ...] = ()
    mode: str = "fallback"
    model: str = "local-rules"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "action_steps": list(self.action_steps),
            "caveats": list(self.caveats),
            "mode": self.mode,
            "model": self.model,
            "metadata": self.metadata,
        }


class ResponseSummarizer:
    """Create concise mixing advice from retrieved forum evidence."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def summarize(
        self,
        query: str,
        results: Sequence[SearchResult],
        presets: Sequence[str] | None = None,
        use_llm: bool = True,
    ) -> MixingSummary:
        if use_llm and self.api_key:
            llm_summary = self._try_llm_summary(query=query, results=results, presets=presets or ())
            if llm_summary is not None:
                return llm_summary
        return self._fallback_summary(query=query, results=results, presets=presets or ())

    def _try_llm_summary(
        self,
        query: str,
        results: Sequence[SearchResult],
        presets: Sequence[str],
    ) -> MixingSummary | None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None

        evidence = "\n".join(
            f"- {result.doc_id} ({result.score:.2f}): {result.text}"
            for result in results[:5]
        )
        preset_text = "\n".join(f"- {preset}" for preset in presets[:5]) or "- Keine Presets"
        prompt = (
            "Du bist ein Mixing Assistant. Antworte auf Deutsch, knapp und praktisch. "
            "Nutze nur die gelieferten Suchtreffer und Preset-Hinweise. "
            "Gib eine kurze Diagnose und 3 konkrete Arbeitsschritte.\n\n"
            f"Problem: {query}\n\n"
            f"Suchtreffer:\n{evidence}\n\n"
            f"Preset-Hinweise:\n{preset_text}"
        )

        try:
            client = OpenAI(api_key=self.api_key)
            if hasattr(client, "responses"):
                response = client.responses.create(
                    model=self.model,
                    input=prompt,
                    temperature=0.2,
                    max_output_tokens=350,
                )
                text = getattr(response, "output_text", "") or ""
            else:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=350,
                )
                text = response.choices[0].message.content or ""
        except Exception:
            return None

        answer = _compact_whitespace(text)
        if not answer:
            return None

        return MixingSummary(
            query=query,
            answer=answer,
            action_steps=_extract_steps(answer) or _fallback_steps(query, results, presets),
            caveats=("LLM-Ausgabe gegen Mix-Kontext validieren.",),
            mode="llm",
            model=self.model,
            metadata={"evidence_count": len(results), "preset_count": len(presets)},
        )

    def _fallback_summary(
        self,
        query: str,
        results: Sequence[SearchResult],
        presets: Sequence[str],
    ) -> MixingSummary:
        top_result = results[0] if results else None
        evidence_text = top_result.text if top_result else "Keine passenden Forumtreffer gefunden."
        answer = (
            f"Für '{query}' zeigt die Knowledge Base vor allem diesen Referenzfall: "
            f"{evidence_text}"
        )
        if presets:
            answer += f" Der wahrscheinlichste erste Preset-Schritt ist: {presets[0]}."

        return MixingSummary(
            query=query,
            answer=answer,
            action_steps=_fallback_steps(query, results, presets),
            caveats=(
                "Regelbasierte Zusammenfassung ohne externes LLM.",
                "Finale Einstellung nach Gehör und Gain-Staging prüfen.",
            ),
            mode="fallback",
            model="local-rules",
            metadata={"evidence_count": len(results), "preset_count": len(presets)},
        )


def _fallback_steps(
    query: str,
    results: Sequence[SearchResult],
    presets: Sequence[str],
) -> tuple[str, ...]:
    steps: list[str] = []
    normalized_query = query.lower()

    if presets:
        steps.append(f"Starte mit: {presets[0]}")
    if any(token in normalized_query for token in ("kick", "bass", "sub")):
        steps.append("Low-End separat prüfen: Kick/Bass solo, dann im Kontext balancieren.")
    if any(token in normalized_query for token in ("s-laut", "scharf", "harsch", "zisch")):
        steps.append("Problemfrequenz eingrenzen und De-Esser oder schmalen EQ-Cut nur im Kontext setzen.")
    if any(token in normalized_query for token in ("vocal", "gesang", "snare", "gitarre")):
        steps.append("Mitteltonbereich mit kleinem EQ-Schritt prüfen, danach Pegel neu ausrichten.")
    if results:
        steps.append(f"Vergleiche gegen Referenzfall {results[0].doc_id} und übernimm nur passende Maßnahmen.")

    if not steps:
        steps.append("Top-Treffer lesen, gemeinsame Frequenzbereiche identifizieren und nur eine Maßnahme pro Durchlauf testen.")
    return tuple(dict.fromkeys(steps[:4]))


def _extract_steps(answer: str) -> tuple[str, ...]:
    lines = [line.strip(" -0123456789.)") for line in answer.splitlines()]
    steps = [line for line in lines if line and len(line) > 10][:4]
    return tuple(steps)


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
