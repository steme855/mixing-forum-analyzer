from __future__ import annotations

from assistant_summary import ResponseSummarizer
from preset_advisor.search import SearchResult


def test_fallback_summary_uses_search_evidence_and_presets() -> None:
    summarizer = ResponseSummarizer(api_key="")
    results = [
        SearchResult(
            doc_id="doc_1",
            text="Kick zu laut im Mix, Fader um 2 dB zurücknehmen.",
            score=0.91,
        )
    ]

    summary = summarizer.summarize(
        query="Kick zu laut",
        results=results,
        presets=["Fader Trim: Kick-Bus um 2-3 dB absenken"],
        use_llm=False,
    )

    assert summary.mode == "fallback"
    assert "Kick zu laut" in summary.answer
    assert "doc_1" in " ".join(summary.action_steps)
    assert summary.metadata["evidence_count"] == 1


def test_fallback_summary_handles_empty_results() -> None:
    summarizer = ResponseSummarizer(api_key="")

    summary = summarizer.summarize(
        query="Delay verschmiert",
        results=[],
        presets=[],
        use_llm=False,
    )

    assert summary.mode == "fallback"
    assert summary.action_steps
    assert summary.metadata["evidence_count"] == 0
