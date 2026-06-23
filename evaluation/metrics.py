"""Core evaluation metrics used by the Mixing Forum Analyzer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class DistributionSummary:
    """Container for similarity or latency summary statistics."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    p10: float
    p90: float
    histogram: tuple[np.ndarray, np.ndarray]

    def as_dict(self) -> dict[str, float | tuple[np.ndarray, np.ndarray]]:
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p10": self.p10,
            "p90": self.p90,
            "histogram": self.histogram,
        }


def _validate_lengths(results: Sequence[Sequence[str]], relevant: Sequence[Iterable[str]]) -> None:
    if len(results) != len(relevant):
        raise ValueError("results and relevant must contain the same number of queries")


def cosine_similarity_distribution(similarities: ArrayLike, bins: int = 15) -> DistributionSummary:
    """Return descriptive stats for cosine similarity scores."""

    if isinstance(similarities, np.ndarray):
        flat = similarities.reshape(-1)
    else:
        flat = np.asarray(list(similarities), dtype=float)

    if flat.size == 0:
        raise ValueError("similarities must contain at least one value")

    histogram = np.histogram(flat, bins=bins, range=(-1.0, 1.0))
    return DistributionSummary(
        mean=float(np.mean(flat)),
        median=float(np.median(flat)),
        std=float(np.std(flat)),
        min=float(np.min(flat)),
        max=float(np.max(flat)),
        p10=float(np.percentile(flat, 10)),
        p90=float(np.percentile(flat, 90)),
        histogram=histogram,
    )


def mean_reciprocal_rank(results: Sequence[Sequence[str]], relevant: Sequence[Iterable[str]]) -> float:
    """Compute MRR given ranked document IDs per query."""

    _validate_lengths(results, relevant)
    reciprocal_ranks: list[float] = []

    for ranked_docs, relevant_docs in zip(results, relevant):
        relevant_set = {doc_id for doc_id in relevant_docs}
        rank = next((idx + 1 for idx, doc_id in enumerate(ranked_docs) if doc_id in relevant_set), 0)
        reciprocal_ranks.append(0.0 if rank == 0 else 1.0 / rank)

    if not reciprocal_ranks:
        return 0.0
    return float(np.mean(reciprocal_ranks))


def inference_latency_summary(latencies_ms: ArrayLike, bins: int = 10) -> DistributionSummary:
    """Summarise inference latency measurements in milliseconds."""

    if isinstance(latencies_ms, np.ndarray):
        arr = latencies_ms.reshape(-1)
    else:
        arr = np.asarray(list(latencies_ms), dtype=float)

    if arr.size == 0:
        raise ValueError("latencies_ms must contain at least one value")

    histogram = np.histogram(arr, bins=bins)
    return DistributionSummary(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p10=float(np.percentile(arr, 10)),
        p90=float(np.percentile(arr, 90)),
        histogram=histogram,
    )


def top_k_accuracy(
    results: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    ks: Sequence[int] = (3, 5),
) -> Mapping[int, float]:
    """Compute top-k accuracy scores for the provided cutoffs."""

    _validate_lengths(results, relevant)
    if not ks:
        raise ValueError("ks must contain at least one cutoff value")

    ks = tuple(sorted(set(ks)))
    total_queries = len(results)
    accuracies = {k: 0 for k in ks}

    for ranked_docs, relevant_docs in zip(results, relevant):
        relevant_set = {doc_id for doc_id in relevant_docs}
        for k in ks:
            head = ranked_docs[:k]
            if any(doc_id in relevant_set for doc_id in head):
                accuracies[k] += 1

    return {k: value / total_queries if total_queries else 0.0 for k, value in accuracies.items()}
