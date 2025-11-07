"""
preset_advisor.core — semantic + keyword backend for Mixing Forum Analyzer
Fully modular, SBERT + TF-IDF hybrid, Keyphrases & Rule-Tags integrated.
"""

from __future__ import annotations
import json, time, re, warnings, importlib
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# Optional heavy imports guarded
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer, util = None, None

DATA_DIR = Path(__file__).parent
CORPUS_PATH = DATA_DIR / "sample_corpus.json"
PRESET_MAP_PATH = DATA_DIR / "preset_map.json"
KEYPHRASES_PATH = DATA_DIR / "keyphrases.json"
RULE_TAGS_PATH = DATA_DIR / "rule_tags.json"

# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------
def _load_json(path: Path, fallback: Any) -> Any:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback

def load_corpus() -> List[str]:
    data = _load_json(CORPUS_PATH, [])
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    # fallback
    return [
        "Snare klingt zu laut und harsch",
        "Kick zu weich, fehlt Punch",
        "Vocals nasal, 800 Hz absenken",
        "Bass maskiert Kick, Sidechain nötig",
        "S-Laute scharf, De-Esser einsetzen",
    ]

CORPUS = load_corpus()

def load_presets() -> Dict[str, Any]:
    return _load_json(PRESET_MAP_PATH, {})

def load_keyphrases() -> Dict[str, List[str]]:
    return _load_json(KEYPHRASES_PATH, {})

def load_rule_tags() -> Dict[str, Dict[str, Any]]:
    return _load_json(RULE_TAGS_PATH, {})

PRESETS = load_presets()
KEYPHRASES = load_keyphrases()
RULE_TAGS = load_rule_tags()

# -----------------------------------------------------------------------------
# Core Models
# -----------------------------------------------------------------------------
def get_tfidf(corpus: List[str]):
    vect = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform(corpus)
    return vect, X

def get_sbert(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SentenceTransformer(model_name, device="cpu")

# -----------------------------------------------------------------------------
# Search utilities
# -----------------------------------------------------------------------------
def expand_query(q: str) -> str:
    LEX = {
        "snare": ["snare", "rimshot"],
        "kick": ["kick", "kickdrum", "bassdrum"],
        "laut": ["laut", "zu laut", "pegel", "clipping"],
        "boxig": ["boxig", "mulmig", "mittig", "muddy"],
        "scharf": ["scharf", "harsch", "zischelig", "harsh"],
        "punch": ["punch", "attack", "anschlag"],
        "vocal": ["vocal", "gesang", "vox"],
        "bass": ["bass", "subbass", "sub-bass"],
    }
    ql = q.lower()
    extra = []
    for head, syns in LEX.items():
        if any(tok in ql for tok in syns + [head]):
            extra += syns + [head]
    return q + (" " + " ".join(sorted(set(extra))) if extra else "")

def boost_score(score: float, text: str) -> float:
    INST = {"snare": 1.15, "kick": 1.10, "vocal": 1.05, "bass": 1.05}
    for w, mult in INST.items():
        if w in text:
            score *= mult
    if re.search(r"\b\d{2,4}\s*(k?hz)\b", text):
        score *= 1.05
    return float(score)

# -----------------------------------------------------------------------------
# Preset suggestions
# -----------------------------------------------------------------------------
def suggest_presets(query: str, presets: Dict[str, Any]) -> List[str]:
    if not presets:
        return []
    q = query.lower()
    recs, seen = [], set()
    for key, lst in presets.items():
        if not isinstance(lst, list):
            continue
        score = max(fuzz.token_set_ratio(q, key), fuzz.partial_ratio(q, key))
        if score >= 70:
            for r in lst:
                if r not in seen:
                    seen.add(r)
                    recs.append(r)
                if len(recs) >= 5:
                    break
    return recs

def match_rule_tags(text: str, rules: Dict[str, Dict[str, Any]]) -> List[str]:
    tl = text.lower()
    hits = []
    for tag, spec in rules.items():
        pats = spec.get("patterns", []) or []
        if any(p in tl for p in pats):
            hits.append(tag)
    return sorted(set(hits))

# -----------------------------------------------------------------------------
# Searches
# -----------------------------------------------------------------------------
def search_tfidf(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    vect, X = get_tfidf(CORPUS)
    q_vec = vect.transform([expand_query(query)])
    sims = cosine_similarity(q_vec, X).flatten()
    idx = sims.argsort()[::-1][:top_k]
    return [{"text": CORPUS[i], "score": float(boost_score(sims[i], CORPUS[i]))} for i in idx]

def search_sbert(query: str, top_k: int = 3, model_name=None) -> List[Dict[str, Any]]:
    model = get_sbert(model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb = model.encode(CORPUS, normalize_embeddings=True, show_progress_bar=False)
    qv = model.encode([expand_query(query)], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = util.cos_sim(qv, emb)[0].cpu().numpy()
    idx = sims.argsort()[::-1][:top_k]
    return [{"text": CORPUS[i], "score": float(boost_score(sims[i], CORPUS[i]))} for i in idx]

def run_search(query: str, backend="sbert", top_k=3) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if backend.lower().startswith("sbert"):
        try:
            results = search_sbert(query, top_k)
            mode = "SBERT"
        except Exception:
            results = search_tfidf(query, top_k)
            mode = "TF-IDF (fallback)"
    else:
        results = search_tfidf(query, top_k)
        mode = "TF-IDF"
    elapsed = (time.perf_counter() - t0) * 1000
    out = {
        "query": query,
        "backend": mode,
        "elapsed_ms": round(elapsed, 2),
        "results": results,
        "presets": suggest_presets(query, PRESETS),
    }
    for r in out["results"]:
        r["tags"] = match_rule_tags(r["text"], RULE_TAGS)
        r["keyphrases"] = KEYPHRASES.get(r["text"], [])
    return out
