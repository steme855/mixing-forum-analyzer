"""
preset_advisor.evaluation_pipeline
----------------------------------
Evaluation & clustering utilities for Mixing Forum Analyzer.

Includes:
- retrieval metrics (Precision@K, Recall@K, MRR, nDCG)
- keyphrase coverage statistics
- rule-tag matching rate
- unsupervised clustering (TF-IDF + cosine)
"""

from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from preset_advisor.core import (
    CORPUS,
    run_search,
    match_rule_tags,
    KEYPHRASES,
    RULE_TAGS,
)

DATA_DIR = Path(__file__).parent
OUT_DIR = DATA_DIR / "metrics"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Basic Metrics
# ---------------------------------------------------------------------
def precision_at_k(results: List[int], relevant: List[int], k: int) -> float:
    if not results or not relevant:
        return 0.0
    retrieved = results[:k]
    hits = len(set(retrieved) & set(relevant))
    return hits / k

def recall_at_k(results: List[int], relevant: List[int], k: int) -> float:
    if not results or not relevant:
        return 0.0
    hits = len(set(results[:k]) & set(relevant))
    return hits / len(relevant)

def mrr(results: List[int], relevant: List[int]) -> float:
    for rank, idx in enumerate(results, 1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(scores: List[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    if not scores:
        return 0.0
    dcg = sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))
    ideal = sum(sorted(scores, reverse=True)[:k][i] / math.log2(i + 2) for i in range(min(k, len(scores))))
    return dcg / ideal if ideal > 0 else 0.0

# ---------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------
def evaluate_queries(queries: List[str], backend="sbert", top_k=5) -> pd.DataFrame:
    """
    Runs evaluation for a set of queries (string list)
    Returns DataFrame with metrics and per-query details.
    """
    rows = []
    for q in queries:
        res = run_search(q, backend=backend, top_k=top_k)
        result_indices = [i for i, _ in enumerate(res["results"])]
        scores = [r["score"] for r in res["results"]]
        rel = [i for i, r in enumerate(res["results"]) if any(tag in r["tags"] for tag in RULE_TAGS.keys())]

        row = {
            "query": q,
            "backend": res["backend"],
            "elapsed_ms": res["elapsed_ms"],
            "precision@3": precision_at_k(result_indices, rel, 3),
            "recall@3": recall_at_k(result_indices, rel, 3),
            "mrr": mrr(result_indices, rel),
            "ndcg@5": ndcg_at_k(scores, 5),
            "avg_score": np.mean(scores) if scores else 0,
            "preset_count": len(res.get("presets", [])),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "eval_results.csv", index=False)
    return df

# ---------------------------------------------------------------------
# Keyphrase & Rule Analysis
# ---------------------------------------------------------------------
def analyze_keyphrases() -> pd.DataFrame:
    """
    Computes coverage stats: how many corpus entries have associated keyphrases or tags.
    """
    data = []
    for text in CORPUS:
        key_count = len(KEYPHRASES.get(text, []))
        tags = match_rule_tags(text, RULE_TAGS)
        data.append({
            "text": text,
            "keyphrases": key_count,
            "tags": len(tags),
            "has_meta": bool(key_count or tags),
        })
    df = pd.DataFrame(data)
    df["keyphrase_density"] = df["keyphrases"] / df["text"].str.split().str.len()
    df.to_csv(OUT_DIR / "keyphrase_stats.csv", index=False)
    return df

# ---------------------------------------------------------------------
# Clustering (TF-IDF + KMeans)
# ---------------------------------------------------------------------
def cluster_corpus(n_clusters: int = 5) -> pd.DataFrame:
    """
    Simple unsupervised clustering for qualitative grouping of corpus posts.
    """
    vect = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1)
    X = vect.fit_transform(CORPUS)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    terms = np.array(vect.get_feature_names_out())
    centroids = km.cluster_centers_.argsort()[:, ::-1]
    top_terms = [[terms[i] for i in centroids[c, :5]] for c in range(n_clusters)]

    df = pd.DataFrame({
        "text": CORPUS,
        "cluster": labels,
        "top_terms": [", ".join(top_terms[label]) for label in labels],
    })
    df.to_csv(OUT_DIR / "clusters.csv", index=False)
    return df

# ---------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------
def run_full_eval(queries: List[str] | None = None, backend="sbert") -> Dict[str, Any]:
    """
    Runs evaluation, keyphrase analysis, and clustering sequentially.
    """
    t0 = time.perf_counter()
    if queries is None:
        queries = [
            "Snare zu laut",
            "Kick weich",
            "S-Laute scharf",
            "Bass maskiert Kick",
        ]
    df_eval = evaluate_queries(queries, backend=backend)
    df_key = analyze_keyphrases()
    df_cluster = cluster_corpus()

    summary = {
        "backend": backend,
        "n_queries": len(queries),
        "mean_precision@3": df_eval["precision@3"].mean(),
        "mean_mrr": df_eval["mrr"].mean(),
        "mean_ndcg@5": df_eval["ndcg@5"].mean(),
        "mean_keyphrases": df_key["keyphrases"].mean(),
        "clusters": df_cluster["cluster"].nunique(),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

if __name__ == "__main__":
    print(json.dumps(run_full_eval(), indent=2, ensure_ascii=False))
