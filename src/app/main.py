"""
Mixing Forum Analyzer — Unified Demo (stabil + features)
- Ruhige Logs (Streamlit-Watcher aus, Torch-Noise gedämpft, Silent-Warmup)
- SBERT lazy-load (mit TF-IDF Fallback)
- spaCy DE robust laden + Diagnose
- Synonym-Expansion, Re-Ranking/Boost, Lemma-Overlap-Erklärungen
- Preset-Map: unterstützt flache Maps und strukturiertes Schema mit Stufen
- CSV/JSON Export, GIF-Vorschau, Enter-to-send, Status-Badges
"""

from __future__ import annotations
import os
import re
import io
import json
import time
import logging
import contextlib
import sys, warnings
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import importlib.util
from collections import defaultdict

from assistant_summary import ResponseSummarizer
from audio_analysis import analyze_wav_bytes
from data_ingestion import load_forum_corpus_texts
from feedback import FeedbackEntry, FeedbackStore
from personal_presets import PersonalPreset, PersonalPresetLibrary
from preset_advisor.search import SearchResult

try:
    from rapidfuzz import fuzz
except ModuleNotFoundError:
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def ratio(a: str, b: str) -> int:
            return int(SequenceMatcher(None, a, b).ratio() * 100)

        @staticmethod
        def partial_ratio(a: str, b: str) -> int:
            if not a or not b:
                return 0
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            shorter = shorter.strip()
            longer = longer.strip()
            if not shorter or not longer:
                return 0
            if len(shorter) >= len(longer):
                return int(SequenceMatcher(None, shorter, longer).ratio() * 100)
            best = 0.0
            window_size = len(shorter)
            for start in range(0, len(longer) - window_size + 1):
                candidate = longer[start : start + window_size]
                best = max(best, SequenceMatcher(None, shorter, candidate).ratio())
            return int(best * 100)

        @staticmethod
        def token_set_ratio(a: str, b: str) -> int:
            tokens_a = set(a.lower().split())
            tokens_b = set(b.lower().split())
            if not tokens_a or not tokens_b:
                return 0
            shared = " ".join(sorted(tokens_a & tokens_b))
            combined_a = " ".join(sorted(tokens_a))
            combined_b = " ".join(sorted(tokens_b))
            if shared:
                return max(
                    int(SequenceMatcher(None, shared, combined_a).ratio() * 100),
                    int(SequenceMatcher(None, shared, combined_b).ratio() * 100),
                )
            return int(SequenceMatcher(None, combined_a, combined_b).ratio() * 100)

    fuzz = _FallbackFuzz()

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Performance & UI Constants
DEFAULT_MAX_RESULTS = 5
MIN_RESULTS = 3
MAX_RESULTS = 10
DEFAULT_INPUT_HEIGHT = 100
STATUS_COLUMNS = 2

# Fuzzy Matching & Scoring
MIN_FUZZY_SCORE = 70
MAX_RECOMMENDATIONS = 5
MAX_SCORED_CANDIDATES = 6

# Model & Embedding Constants  
DEFAULT_EMBEDDING_DIM = 384
LARGE_EMBEDDING_DIM = 512
MODEL_DOWNLOAD_SIZE_MB = 80

# File Processing
MAX_SPACY_DISPLAY_ROWS = 200
DEFAULT_NGRAM_RANGE = (1, 2)
MIN_DOCUMENT_FREQUENCY = 1

# Frequency & Audio Constants
DEFAULT_VOCAL_FREQ_HZ = 800
SIBILANCE_FREQ_RANGE = "6-8 kHz"
DEESSER_REDUCTION_DB = 4
DEESSER_RANGE_DB = 6

# Timing & Performance
TORCH_WARMUP_LINES = 30
MAX_LINE_LENGTH = 120

# Scoring Boosts
INSTRUMENT_BOOST = {"snare": 1.15, "kick": 1.10, "vocal": 1.05, "bass": 1.05}
FREQUENCY_PATTERN_BOOST = 1.05

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_CORPUS_PATH = DATA_DIR / "sample_corpus.json"
PRESET_MAP_PATH = DATA_DIR / "preset_map.json"
KEYPHRASES_PATH = DATA_DIR / "keyphrases.json"
RULE_TAGS_PATH = DATA_DIR / "rule_tags.json"
FEEDBACK_STORE = FeedbackStore(DATA_DIR / "feedback" / "feedback.jsonl")
SUMMARY_SERVICE = ResponseSummarizer()
PERSONAL_PRESETS = PersonalPresetLibrary(DATA_DIR / "personal_presets")


def _resolve_data_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return DATA_DIR / candidate



# -----------------------------
# Environment & logging hygiene
# -----------------------------
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_FILE_WATCHER", "false")
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
os.environ.pop("TORCH_LOGS", None)  # invalid values crash PyTorch
os.environ.setdefault("PYTORCH_SHOW_CPP_STACKTRACES", "0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("DEBUG_STREAMLIT", "0")

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

def _quiet_torch_logs() -> None:
    warnings.filterwarnings("ignore", message=r".*torch\.classes.*")
    warnings.filterwarnings("ignore", message=r".*Triton.*", category=UserWarning)
    logging.getLogger("torch").setLevel(logging.ERROR)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
_quiet_torch_logs()

@contextlib.contextmanager
def _silence_streams():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield

def _warmup_torch_silently() -> None:
    import importlib
    with _silence_streams():
        try:
            importlib.import_module("torch")
        except Exception:
            pass
_warmup_torch_silently()

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Mixing Forum Analyzer", layout="wide")
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
st.title("🎛️ Mixing Forum Analyzer — Woche 2 · Keyphrases & Tags")
st.write("Semantische Suche (SBERT/TF‑IDF) · Preset‑Vorschläge · spaCy POS/Adjektive · **Week 2:** Keyphrases & Rule‑Tags + Export.")

# Header badges (Modus/Timing)
status_cols = st.columns(2)
with status_cols[0]:
    if st.session_state.get("mode_label"):
        mode = st.session_state["mode_label"]
        color = "green" if "SBERT" in mode else "blue"
        st.markdown(f"<span style='color:{color}; font-weight:bold'>● {mode}</span>", unsafe_allow_html=True)
with status_cols[1]:
    if st.session_state.get("elapsed_ms"):
        ms = st.session_state["elapsed_ms"]
        st.markdown(f"<span style='color:gray'>⏱ {ms:.0f} ms</span>", unsafe_allow_html=True)
if "mode_label" in st.session_state and "elapsed_ms" in st.session_state:
    st.caption(f"Modus: {st.session_state['mode_label']} · letzte Suche: {st.session_state['elapsed_ms']:.0f} ms")

# -----------------------------
# Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_corpus(path: Union[str, Path] = SAMPLE_CORPUS_PATH) -> List[str]:
    """
    Lädt das Forum-Korpus aus der neuen Ingestion-Schicht.

    Der `path`-Parameter bleibt aus Kompatibilitätsgründen erhalten,
    wird aber zugunsten der zentralen Data-Ingestion-Pipeline ignoriert.
    """
    try:
        texts = load_forum_corpus_texts(DATA_DIR)
        if texts:
            return texts
    except Exception:
        pass
    return [
        "Die Snare ist zu laut und harsch",
        "Kick zu weich, es fehlt der Punch",
        "Vocals klingen nasal, 800 Hz absenken",
        "Bass maskiert die Kick, Sidechain nötig",
        "S-Laute sind scharf, De-Esser einsetzen",
    ]

def _normalize_preset_map(pm: Any) -> Dict[str, Any]:
    """Make preset map robust to different shapes.
    Accepts:
    - structured dict with "entries" (kept as is)
    - flat dict {"snare":[...], ...} (kept as is)
    - list of dicts (structured entries) -> wrap into {"entries": [...]} 
    - list of strings (generic presets) -> wrap into {"_generic": [...]}
    Otherwise -> return empty structured schema.
    """
    # structured dict already
    if isinstance(pm, dict):
        if "entries" in pm and isinstance(pm.get("entries"), list):
            return pm
        return pm  # treat any other dict as flat map
    # list inputs
    if isinstance(pm, list):
        if all(isinstance(x, dict) and ("key" in x or "actions" in x) for x in pm):
            return {"schema_version": "1.0", "entries": pm}
        if all(isinstance(x, str) for x in pm):
            return {"_generic": pm}
    # fallback
    return {"schema_version": "1.0", "entries": []}

@st.cache_data(show_spinner=False)
def load_preset_map(path: Union[str, Path] = PRESET_MAP_PATH):
    resolved_path = _resolve_data_path(path)
    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return _normalize_preset_map({"schema_version": "1.0", "entries": []})
            return _normalize_preset_map(json.loads(txt))
    except Exception as e:
        st.warning(f"Preset-Map konnte nicht geladen werden ({e}). Fallback aktiv.")
        return _normalize_preset_map({"schema_version": "1.0", "entries": []})


corpus = load_corpus()
preset_map = load_preset_map()

# --- Week 2: Keyphrases & Rule Tags loaders ---
@st.cache_data(show_spinner=False)
def load_keyphrases(path: Union[str, Path] = KEYPHRASES_PATH) -> Dict[str, List[str]]:
    """
    Robust loader. Supports:
    - dict[str -> list[str]]
    - list[list[str]] aligned to corpus indices
    - list[str] (single flat list) -> returned under key "_global"
    """
    resolved_path = _resolve_data_path(path)
    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # normalize keys to str to avoid json numeric-vs-str surprises
            return {str(k): list(v) if isinstance(v, list) else [str(v)] for k, v in data.items()}
        if isinstance(data, list):
            if all(isinstance(x, list) for x in data):
                return {str(i): list(xs) for i, xs in enumerate(data)}
            if all(isinstance(x, str) for x in data):
                return {"_global": list(data)}
        return {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.warning(f"Keyphrases konnten nicht geladen werden ({e}).", icon="⚠️")
        return {}

@st.cache_data(show_spinner=False)
def load_rule_tags(path: Union[str, Path] = RULE_TAGS_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Expected shape:
    {
      "sibilance": {"patterns": ["s-laut","zisch","sibil"], "tip": "De-Esser 6–8 kHz"},
      "kick_punch": {"patterns": ["kick","punch"], "tip": "Transient/Parallel-Comp"}
    }
    """
    resolved_path = _resolve_data_path(path)
    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            norm: Dict[str, Dict[str, Any]] = {}
            for tag, spec in data.items():
                if isinstance(spec, dict):
                    pats = spec.get("patterns", [])
                    tip = spec.get("tip", "")
                    # normalize
                    if isinstance(pats, str):
                        pats = [pats]
                    pats = [str(p).lower() for p in pats if isinstance(p, (str, int, float))]
                    norm[tag] = {"patterns": pats, "tip": str(tip)}
            return norm
        return {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.warning(f"Rule-Tags konnten nicht geladen werden ({e}).", icon="⚠️")
        return {}

def match_rule_tags(text: str, rules: Dict[str, Dict[str, Any]]) -> List[str]:
    """Very lightweight keyword/substring matcher (case-insensitive)."""
    tl = text.lower()
    hits: List[str] = []
    for tag, spec in rules.items():
        pats = spec.get("patterns", []) or []
        if any(p in tl for p in pats):
            hits.append(tag)
    # keep order stable & unique
    seen = set()
    out: List[str] = []
    for t in hits:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

keyphrases_index = load_keyphrases()
rule_tags = load_rule_tags()

@st.cache_resource(show_spinner=False)
def build_tfidf(corpus_texts: List[str]):
    """
    Erstellt TF-IDF Vectorizer und transformiert das Korpus.

    Konfiguriert für deutsche Audio-Engineering Texte mit Bigramm-Support
    und case-insensitive Verarbeitung.

    Args:
        corpus_texts: Liste der zu vektorisierenden Texte

    Returns:
        Tuple[TfidfVectorizer, sparse matrix]: Fitted Vectorizer und Matrix
    """
    vect = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    X = vect.fit_transform(corpus_texts)
    return vect, X

# -----------------------------
# SBERT (lazy)
# -----------------------------
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource(show_spinner=False)
def load_sbert(model_name: str):
    """
    Lädt ein Sentence-BERT Modell für semantische Suche.

    Verwendet CPU-Device und unterdrückt Warnings während des Ladens.
    Gibt None zurück bei Fehlern (z.B. Triton-Konflikte).

    Args:
        model_name: HuggingFace Model-Name (z.B. paraphrase-multilingual-MiniLM-L12-v2)

    Returns:
        SentenceTransformer | None: Geladenes Modell oder None bei Fehlern
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _silence_streams():
                from sentence_transformers import SentenceTransformer  # type: ignore
                return SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        st.warning(
            "SBERT konnte nicht geladen werden – Fallback auf TF-IDF.\n\n"
            f"Fehler: {e}\n\n"
            "Hinweis macOS/CPU: Wenn **triton** installiert ist → `pip uninstall -y triton` "
            "und Torch pinnen: `pip install torch==2.2.2`.",
            icon="⚠️",
        )
        return None

@st.cache_data(show_spinner=False)
def embed_corpus(model_name: str, docs: List[str]):
    """
    Erstellt normalisierte Embeddings für eine Dokumentensammlung.

    Args:
        model_name: Name des SBERT-Modells
        docs: Liste der zu embedded Dokumente

    Returns:
        np.ndarray | None: Normalisierte Embedding-Matrix oder None bei Fehlern
    """
    model = load_sbert(model_name)
    if model is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _silence_streams():
            return model.encode(docs, normalize_embeddings=True)

def _detect_triton_conflict() -> bool:
    """
    Prüft ob das separate Triton-Paket installiert ist.

    Triton kann mit PyTorch konfligieren und SBERT-Loading blockieren.

    Returns:
        bool: True wenn Triton-Konflikt erkannt
    """
    try:
        return importlib.util.find_spec("triton") is not None
    except Exception:
        return False

def _render_triton_warning(sidebar: bool = True) -> None:
    msg = (
        "Triton erkannt – das separate Paket kollidiert oft mit PyTorch und blockiert SBERT.\n\n"
        "**Lösung (im aktiven venv):**\n"
        "`pip uninstall -y triton`  \\n"
        "`pip install --no-cache-dir \"torch==2.2.2\" sentence-transformers==3.0.1`"
    )
    (st.sidebar if sidebar else st).warning(msg, icon="⚠️")

# -----------------------------
# spaCy (robust + Diagnose)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_spacy(model_name: str = "de_core_news_sm"):
    """Load the German spaCy model with installation fallback."""

    try:
        import spacy
        import spacy.util as spacy_util  # type: ignore
    except Exception as exc:
        st.session_state["spacy_error"] = f"spaCy import failed: {exc}  [python: {sys.executable}]"
        return None

    def _try_load() -> Optional["spacy.Language"]:
        try:
            return spacy.load(model_name)
        except OSError:
            pass
        except Exception as load_exc:
            st.session_state["spacy_error"] = f"spaCy load failed: {load_exc}"
            return None
        package_name = model_name.replace("-", "_")
        try:
            module = importlib.import_module(package_name)
            if hasattr(module, "load"):
                return module.load()  # type: ignore[call-arg]
        except Exception:
            return None
        return None

    nlp = _try_load()
    if nlp is not None:
        st.session_state.pop("spacy_error", None)
        return nlp

    try:
        if getattr(spacy_util, "is_package", None) and spacy_util.is_package(model_name):
            nlp = _try_load()
            if nlp is not None:
                st.session_state.pop("spacy_error", None)
                return nlp
    except Exception:
        pass

    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as install_err:
        msg = install_err.stderr.strip() or install_err.stdout.strip() or str(install_err)
        st.session_state["spacy_error"] = f"spaCy download failed: {msg}"
        return None
    except Exception as install_exc:
        st.session_state["spacy_error"] = f"spaCy download error: {install_exc}"
        return None

    nlp = _try_load()
    if nlp is None:
        st.session_state["spacy_error"] = f"spaCy model '{model_name}' not available after download."
        return None

    st.session_state.pop("spacy_error", None)
    return nlp

def spacy_diagnostics() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import spacy
        info["spacy_version"] = getattr(spacy, "__version__", "unknown")
        info["spacy_path"] = str(Path(spacy.__file__).parent)
        try:
            import spacy.util as spacy_util  # type: ignore
            info["is_package(de_core_news_sm)"] = bool(spacy_util.is_package("de_core_news_sm"))
        except Exception:
            info["is_package(de_core_news_sm)"] = "n/a"
    except Exception as e:
        info["spacy_import_error"] = str(e)
    info["find_spec(de_core_news_sm)"] = bool(importlib.util.find_spec("de_core_news_sm"))
    try:
        import de_core_news_sm  # type: ignore
        info["de_core_news_sm_importable"] = True
        try:
            info["de_core_news_sm_path"] = str(Path(de_core_news_sm.__file__).parent)  # type: ignore
        except Exception:
            pass
    except Exception as e:
        info["de_core_news_sm_importable"] = False
        info["de_core_news_sm_error"] = str(e)
    info["python_executable"] = sys.executable
    return info

# -----------------------------
# Preset-Map (flat + structured)
# -----------------------------
def is_structured_schema(pm: Dict[str, Any]) -> bool:
    return isinstance(pm, dict) and "entries" in pm and isinstance(pm["entries"], list)

def render_action(a: Dict[str, Any]) -> str:
    t = a.get("type", "step")
    if t == "eq":
        band = a.get("band", "bell"); freq = a.get("freq_hz", "?"); gain = a.get("gain_db", 0); q = a.get("q", 1.0)
        return f"EQ: {band} @ {freq} Hz, {gain} dB, Q {q}"
    if t == "de-esser":
        pos = a.get("position", ""); red = a.get("reduction_db", 2); extra = f" ({pos.replace('_',' ')})" if pos else ""
        return f"De-Esser{extra}: {red} dB"
    if t == "dynamic_eq":
        freq = a.get("freq_hz", "?"); thr = a.get("threshold_db", "?"); ratio = a.get("ratio", "?")
        return f"Dyn-EQ @ {freq} Hz, Th {thr} dB, Ratio {ratio}"
    if t == "transient":
        mode = a.get("mode", "attack_plus"); amt = a.get("amount", 0.3)
        return f"Transient ({mode}): {amt}"
    if t == "parallel_comp":
        ratio = a.get("ratio", 4.0); blend = a.get("blend", 0.3)
        return f"Parallel Comp: Ratio {ratio}:1, Blend {int(blend*100)}%"
    if t == "clipper":
        amt = a.get("amount_db", 1.0); return f"Clipper: {amt} dB"
    if t == "sidechain":
        duck = a.get("duck_db", 2.0); atk = a.get("attack_ms", 10); rel = a.get("release_ms", 120)
        return f"Sidechain: Duck {duck} dB, Attack {atk} ms, Release {rel} ms"
    if t == "hp_filter_sc":
        freq = a.get("freq_hz", 120); return f"SC-HPF @ {freq} Hz"
    return ", ".join([f"{k}={v}" for k, v in a.items()])

def suggest_presets_flat(text: str, pm: Any) -> List[str]:
    if not text or not pm:
        return []
    # Accept alternate shapes
    if isinstance(pm, list):
        # treat as generic list of suggestions
        return list(dict.fromkeys(pm))[:5]
    if isinstance(pm, dict) and isinstance(pm.get("_generic"), list):
        return list(dict.fromkeys(pm["_generic"]))[:5]
    q = text.lower().strip()
    SYNS = {
        "snare": ["snare", "rimshot"],
        "kick": ["kick", "kickdrum", "bassdrum"],
        "weich": ["weich", "soft", "dull"],
        "boxig": ["boxig", "mulmig", "mittig", "muddy"],
        "scharf": ["scharf", "harsch", "zischelig", "harsh"],
        "punch": ["punch", "attack", "anschlag"],
        "s-laut": ["s-laut", "s-laute", "sibilanz", "sibilanzen", "zischlaut"],
        "bass": ["bass", "subbass", "sub-bass"],
        "maskiert": ["maskiert", "verdeckt", "überlagert"],
    }
    for syns in SYNS.values():
        if any(s in q for s in syns):
            q += " " + " ".join(syns)
    candidates = [k for k in pm.keys() if not str(k).startswith("_") and str(k) not in {"schema_version", "entries"}]
    if not candidates:
        return []
    scored = []
    for key in candidates:
        k = key.lower()
        s1 = fuzz.token_set_ratio(q, k)
        s2 = fuzz.partial_ratio(q, k)
        score = max(s1, s2)
        if score >= MIN_FUZZY_SCORE:
            scored.append((score, key))
    scored.sort(reverse=True)
    recs, seen = [], set()
    for _, key in scored[:6]:
        for r in pm.get(key, []):
            if r not in seen:
                seen.add(r)
                recs.append(r)
            if len(recs) >= 5:
                break
        if len(recs) >= 5:
            break
    return recs

def suggest_presets_structured(text: str, pm: Dict[str, Any], level: str) -> List[str]:
    if not text or not is_structured_schema(pm):
        return []
    q = text.lower().strip()
    entries = pm.get("entries", [])
    scored = []
    for e in entries:
        keys = [e.get("key", "").lower()] + [a.lower() for a in e.get("aliases", [])]
        keys = [k for k in keys if k]
        if not keys:
            continue
        best = 0
        for k in keys:
            s1 = fuzz.token_set_ratio(q, k)
            s2 = fuzz.partial_ratio(q, k)
            best = max(best, s1, s2)
        if best >= 70:
            scored.append((best, e))
    scored.sort(reverse=True)
    recs = []
    for _, e in scored[:6]:
        tier = (e.get("actions", {}) or {}).get(level, [])
        for a in tier:
            recs.append(render_action(a))
            if len(recs) >= 5:
                break
        if len(recs) >= 5:
            break
    return recs

def suggest_presets(text: str, level: str) -> List[str]:
    if is_structured_schema(preset_map):
        return suggest_presets_structured(text, preset_map, level)
    return suggest_presets_flat(text, preset_map)

def tooltip_for(rec_str: str) -> str:
    tmap = {
        "de-esser": "De-Esser: 4–6 dB bei 6–8 kHz",
        "eq:": "EQ: gezielte Bänder, Gain moderat, Q nach Bedarf",
        "dynamic_eq": "Dyn-EQ: nur bei Überschreitung absenken",
        "dyn-eq": "Dyn-EQ: nur bei Überschreitung absenken",
        "transient": "Transient: Attack erhöhen für mehr Punch",
        "parallel comp": "Parallel Comp: Blend 20–40%",
        "clipper": "Clipper: sanftes Limiting, 1–2 dB",
        "sidechain": "Sidechain: Kick duckt Bass 2–4 dB",
        "sc-hpf": "SC-HPF: SC unter 80–150 Hz filtern",
    }
    s = rec_str.lower()
    for k, v in tmap.items():
        if k in s:
            return v
    return "Preset-Hinweis"

# -----------------------------
# Sidebar
# -----------------------------
send_on_enter = st.sidebar.checkbox("Senden mit Enter (einzeilig)", value=False)
use_semantics = st.sidebar.toggle("Semantische Embeddings (SBERT)", value=False,
                                  help="Aktivieren lädt ein SBERT-Modell (~80 MB). Aus = TF-IDF.")
model_options = {
    "Multilingual MiniLM (kompakt, 384d)": DEFAULT_MODEL,
    "Multilingual DistilUSE (größer, 512d)": "sentence-transformers/distiluse-base-multilingual-cased-v2",
}
if use_semantics:
    sel = st.sidebar.selectbox("SBERT-Modell", list(model_options.keys()), index=0)
    model_name = model_options[sel]
else:
    model_name = DEFAULT_MODEL

if use_semantics and _detect_triton_conflict():
    _render_triton_warning(sidebar=True)

k = st.sidebar.slider("Anzahl Treffer", min_value=MIN_RESULTS, max_value=MAX_RESULTS, value=DEFAULT_MAX_RESULTS)
show_explanations = st.sidebar.checkbox("Erklärungen anzeigen (Lemma-Overlap)", value=True)
show_overlap_chart = st.sidebar.checkbox("Mini-Bar-Chart anzeigen", value=True)
show_ai_summary = st.sidebar.checkbox("Assistant-Zusammenfassung", value=True)
use_llm_summary = st.sidebar.checkbox(
    "LLM nutzen, falls API-Key vorhanden",
    value=False,
    help="Ohne OPENAI_API_KEY nutzt die App automatisch eine lokale Regel-Zusammenfassung.",
)

# Demo-GIF
with st.sidebar.expander("🎥 Demo (30 Sek.)", expanded=False):
    gif_candidates = [
        Path(__file__).parent / "app" / "assets" / "demo.gif",  # app at project root
        Path(__file__).parent / "assets" / "demo.gif",          # typical: app/app.py
        Path(__file__).parent.parent / "demo.gif",              # repo root
        Path.cwd() / "app" / "assets" / "demo.gif",
        Path.cwd() / "assets" / "demo.gif",
    ]
    gif_path = next((p for p in gif_candidates if p.exists()), None)
    if gif_path:
        try:
            st.image(str(gif_path), use_container_width=True)
        except TypeError:
            st.image(str(gif_path), use_column_width=True)
        st.download_button("GIF herunterladen", data=gif_path.read_bytes(), file_name="demo.gif", mime="image/gif")
        st.caption(f"GIF gefunden: {gif_path}")
    else:
        st.caption("Lege `app/assets/demo.gif` oder `./demo.gif` an, um die Vorschau zu sehen.")

# -----------------------------
# Query-Eingabe & Buttons
# -----------------------------
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

examples = [
    ("Snare zu laut", "Snare zu laut"),
    ("Kick weich / Punch", "Kick zu weich, mehr Punch"),
    ("S-Laute scharf", "S-Laute zu scharf, De-Esser?"),
    ("Bass maskiert Kick", "Bass maskiert Kick, Sidechain einrichten"),
    ("✖︎ Clear", ""),
]
for (label, val), col in zip(examples, st.columns(5)):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["query_text"] = val
            st.session_state["trigger_search"] = True
            st.rerun()

def _trigger_search_cb(): st.session_state["trigger_search"] = True
label_txt = "🔎 Beschreibe dein Mixing-Problem (z. B. 'Snare zu boxig, mehr Punch auf Kick')"
placeholder_txt = st.session_state.get("query_text", "") or "z. B. 'Snare zu boxig, mehr Punch auf Kick'"

if send_on_enter:
    st.text_input(label_txt, key="query_text", placeholder=placeholder_txt, on_change=_trigger_search_cb)
    st.caption("↩️ **Enter** startet sofort die Suche (einzeiliges Feld).")
else:
    st.text_area(label_txt, key="query_text", placeholder=placeholder_txt, height=DEFAULT_INPUT_HEIGHT)
    st.caption("⌨️ **Ctrl/⌘ + Enter** – danach auf *Ähnliche Posts finden* klicken.")

preset_level = st.selectbox(
    "Preset-Intensität",
    options=["konservativ", "standard", "aggressiv"],
    index=1,
    help="Wirkt bei strukturiertem Mapping (conservative/standard/aggressive)."
)
st.caption("ℹ️ **Preset-Intensität**: konservativ = dezent · standard = ausgewogen · aggressiv = deutlich.")

col1, col2 = st.columns(2)

# -----------------------------
# Left: Suche + Presets + Erklärungen
# -----------------------------
with col1:
    st.subheader("Semantische Suche")
    search_clicked = st.button("Ähnliche Posts finden") or st.session_state.pop("trigger_search", False)

    # Domain lexicon for query expansion
    LEX = {
        "snare": ["snare", "rimshot"],
        "kick": ["kick", "kickdrum", "bassdrum"],
        "laut": ["laut", "zu laut", "pegel hoch", "clipping", "übersteuern"],
        "boxig": ["boxig", "mulmig", "mittig", "muddy"],
        "scharf": ["scharf", "harsch", "zischelig", "harsh"],
        "punch": ["punch", "attack", "anschlag"],
        "vocal": ["vocal", "gesang", "vox"],
        "bass": ["bass", "subbass", "sub-bass"],
    }
    def expand_query(q: str) -> str:
        ql = q.lower(); extra = []
        for head, syns in LEX.items():
            if any(tok in ql for tok in syns + [head]):
                extra += syns + [head]
        return q + (" " + " ".join(sorted(set(extra))) if extra else "")

    INST_WORDS = {"snare": 1.15, "kick": 1.10, "vocal": 1.05, "bass": 1.05}
    def boost_score(base: float, text_l: str) -> float:
        b = base
        for w, mult in INST_WORDS.items():
            if w in text_l:
                b *= mult
        if re.search(r"\b\d{2,4}\s*(k?hz)\b", text_l):
            b *= 1.05
        return float(b)

    if search_clicked:
        q = st.session_state.get("query_text", "").strip()
        if not q:
            st.warning("Bitte eine Query eingeben.")
        else:
            if not corpus:
                st.error("Korpus ist leer. Lege `data/sample_corpus.json` an oder fülle ihn mit Beispiel-Posts.")
                st.stop()
            q_expanded = expand_query(q)
            t0 = time.perf_counter()
            if use_semantics:
                emb = embed_corpus(model_name, corpus)
                if emb is not None:
                    model = load_sbert(model_name)
                    with _silence_streams():
                        qv = model.encode([q_expanded], normalize_embeddings=True)
                    sims = (qv @ emb.T).ravel()
                    mode_label = "SBERT (semantisch)"
                else:
                    vect, X = build_tfidf(corpus)
                    sims = (vect.transform([q_expanded]) @ X.T).toarray().ravel()
                    mode_label = "TF-IDF (lexikalisch)"
            else:
                vect, X = build_tfidf(corpus)
                sims = (vect.transform([q_expanded]) @ X.T).toarray().ravel()
                mode_label = "TF-IDF (lexikalisch)"
            elapsed_ms = (time.perf_counter() - t0) * 1000
            st.session_state["mode_label"] = mode_label
            st.session_state["elapsed_ms"] = elapsed_ms

            scored = [(boost_score(float(s), corpus[i].lower()), i) for i, s in enumerate(sims)]
            scored.sort(reverse=True)
            k_eff = min(k, len(scored))
            top_idx = [i for _, i in scored[:k_eff]]

            df = pd.DataFrame({"Ähnlichkeit": [round(scored[j][0], 3) for j in range(k_eff)],
                               "Post": [corpus[i] for i in top_idx]})
            search_results = [
                SearchResult(doc_id=f"corpus_{i}", text=corpus[i], score=float(scored[r][0]))
                for r, i in enumerate(top_idx)
            ]
            st.subheader(f"Top {k_eff} ähnliche Posts")
            st.dataframe(df, use_container_width=True)
            st.caption(f"⏱️ {mode_label} · Berechnungszeit: {elapsed_ms:.0f} ms")

            # CSV/JSON
            st.download_button("CSV downloaden", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="last_query_results.csv", mime="text/csv")
            results_json = {
                "schema_version": "1.0",
                "query": q,
                "expanded_query": q_expanded,
                "mode": mode_label,
                "elapsed_ms": round(elapsed_ms, 1),
                "results": [
                    {"rank": r + 1, "post_index": int(i), "post": corpus[i], "score": float(round(scored[r][0], 3))}
                    for r, i in enumerate(top_idx)
                ],
            }
            # Presets
            st.subheader("🔧 Vorschläge (Preset/Chain)")
            recs = suggest_presets(q, level={"konservativ": "conservative",
                                             "standard": "standard",
                                             "aggressiv": "aggressive"}[preset_level])
            if recs:
                for r in recs:
                    st.markdown(f"- <span title=\"{tooltip_for(r)}\">{r}</span>", unsafe_allow_html=True)
                results_json["presets"] = recs
                st.caption("Bewege den Mauszeiger über einen Vorschlag für kurze Hinweise.")
            else:
                results_json["presets"] = []
                st.caption("Keine direkten Vorschläge – Formulierung verfeinern (z. B. 'snare boxig', 'bass maskiert kick').")

            if show_ai_summary:
                summary = SUMMARY_SERVICE.summarize(
                    query=q,
                    results=search_results,
                    presets=recs,
                    use_llm=use_llm_summary,
                )
                results_json["summary"] = summary.to_record()
                st.subheader("Assistant-Zusammenfassung")
                st.write(summary.answer)
                if summary.action_steps:
                    st.markdown("**Nächste Mixing-Schritte**")
                    for step in summary.action_steps:
                        st.markdown(f"- {step}")
                st.caption(f"Summary-Modus: {summary.mode} · Modell: {summary.model}")

            st.download_button("JSON downloaden",
                               data=json.dumps(results_json, ensure_ascii=False, indent=2).encode("utf-8"),
                               file_name="last_query_results.json", mime="application/json")

            with st.form("feedback_form"):
                st.subheader("Feedback")
                feedback_rating = st.slider("Qualität der Antwort", min_value=1, max_value=5, value=4)
                feedback_comment = st.text_area("Kommentar", placeholder="Was war hilfreich oder falsch?", height=80)
                feedback_type = st.selectbox("Feedback-Typ", ["search", "preset", "summary"], index=0)
                submitted = st.form_submit_button("Feedback speichern")
                if submitted:
                    try:
                        entry = FEEDBACK_STORE.append(
                            FeedbackEntry(
                                query=q,
                                rating=feedback_rating,
                                comment=feedback_comment,
                                result_doc_ids=tuple(result.doc_id for result in search_results),
                                feedback_type=feedback_type,
                                session_id=st.session_state["session_id"],
                                metadata={
                                    "mode": mode_label,
                                    "elapsed_ms": round(elapsed_ms, 1),
                                    "preset_level": preset_level,
                                },
                            )
                        )
                        st.success(f"Feedback gespeichert: {entry.entry_id}")
                    except ValueError as exc:
                        st.error(str(exc))

            # --- Week 2: Keyphrases & Tags (Expander) ---
            with st.expander("🔖 Keyphrases & Tags (Week 2)", expanded=False):
                if not top_idx:
                    st.caption("Keine Treffer vorhanden – zuerst eine Suche ausführen.")
                else:
                    # build per-result view
                    rows_kt = []
                    for rank, idx in enumerate(top_idx, 1):
                        post_text = corpus[idx]
                        # keyphrases: support dict keyed by index or by text fallback
                        kp: List[str] = []
                        if str(idx) in keyphrases_index:
                            kp = list(keyphrases_index.get(str(idx), []))
                        elif post_text in keyphrases_index:
                            kp = list(keyphrases_index.get(post_text, []))
                        elif "_global" in keyphrases_index:
                            kp = list(keyphrases_index.get("_global", []))

                        # rule tags
                        tags = match_rule_tags(post_text, rule_tags)

                        # render
                        st.markdown(f"**#{rank}** {post_text}")
                        if kp:
                            st.markdown("• **Keyphrases:** " + ", ".join(kp))
                        else:
                            st.markdown("• **Keyphrases:** —")
                        if tags:
                            # attach small tooltips if available
                            tag_bits = []
                            for t in tags:
                                tip = (rule_tags.get(t, {}) or {}).get("tip", "")
                                if tip:
                                    tag_bits.append(f"<span title=\"{tip}\">{t}</span>")
                                else:
                                    tag_bits.append(t)
                            st.markdown("• **Tags:** " + ", ".join(tag_bits), unsafe_allow_html=True)
                        else:
                            st.markdown("• **Tags:** —")
                        st.markdown("---")

                    # Optional: tiny cluster stub (term-based label)
                    try:
                        # derive a rough "cluster label" from the top results by TF-IDF top terms
                        vect_tmp = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1)
                        X_tmp = vect_tmp.fit_transform([corpus[i] for i in top_idx])
                        terms = np.asarray(vect_tmp.get_feature_names_out())
                        scores = np.asarray(X_tmp.sum(axis=0)).ravel()
                        top_terms = terms[np.argsort(-scores)[:5]]
                        cluster_label = ", ".join(top_terms.tolist())
                    except Exception:
                        cluster_label = ""

                    # Export analysis_week2.json
                    analysis = {
                        "schema_version": "2.0",
                        "query": q,
                        "expanded_query": q_expanded,
                        "mode": mode_label,
                        "elapsed_ms": round(elapsed_ms, 1),
                        "cluster_label": cluster_label,
                        "results": [
                            {
                                "rank": r+1,
                                "post_index": int(i),
                                "post": corpus[i],
                                "score": float(round(scored[r][0], 3)),
                                "keyphrases": keyphrases_index.get(str(i), keyphrases_index.get(corpus[i], keyphrases_index.get("_global", []))),
                                "tags": match_rule_tags(corpus[i], rule_tags),
                            }
                            for r, i in enumerate(top_idx)
                        ],
                    }
                    st.download_button(
                        "Week 2: analysis_week2.json downloaden",
                        data=json.dumps(analysis, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="analysis_week2.json",
                        mime="application/json",
                    )
                    if not keyphrases_index and not rule_tags:
                        st.caption("Hinweis: Lege optional `data/keyphrases.json` und `data/rule_tags.json` an, um Inhalte hier zu sehen.")

            # Erklärungen (spaCy)
            if show_explanations:
                nlp = load_spacy()
                if nlp is not None:
                    q_lem = {t.lemma_.lower() for t in nlp(q) if not (t.is_stop or t.is_punct or t.like_num)}
                    st.caption("Warum diese Treffer? (Lemma-Overlap)")
                    overlap_counts, labels = [], []
                    for rank, i in enumerate(top_idx, 1):
                        d_lem = {t.lemma_.lower() for t in nlp(corpus[i]) if not (t.is_stop or t.is_punct or t.like_num)}
                        ov_set = q_lem & d_lem
                        st.markdown(f"**#{rank}** Overlap: _{', '.join(sorted(ov_set)) or '—'}_")
                        overlap_counts.append(len(ov_set)); labels.append(f"#{rank}")
                    if show_overlap_chart:
                        try:
                            chart_df = pd.DataFrame({"Treffer": labels, "Overlap": overlap_counts}).set_index("Treffer")
                            if chart_df["Overlap"].sum() > 0:
                                st.bar_chart(chart_df)
                        except Exception:
                            pass
                else:
                    st.caption("Erklär-Overlay benötigt spaCy DE.")

# -----------------------------
# Right: spaCy inspection panel
# -----------------------------
with col2:
    st.subheader("spaCy Analyse (DE) — Adjektive & Lemmata")
    spacy_text = st.text_area("Text für spaCy (optional)",
                              value="Die Snare klingt trocken und etwas hart, vielleicht mehr Raumanteil.",
                              key="spacy_text", height=100)
    nlp = load_spacy()
    if not nlp:
        st.info("spaCy DE nicht verfügbar – installiere mit: `python -m spacy download de_core_news_sm`.")
        if "spacy_error" in st.session_state:
            st.caption("Fehler: " + st.session_state["spacy_error"])
        with st.expander("spaCy Diagnose", expanded=False):
            diag = spacy_diagnostics()
            st.write(diag)
            st.code("Aktives Python:\n" + diag.get("python_executable", ""))
            st.code("Install:\npython -m pip show spacy\necho $(python -c \"import sys;print(sys.executable)\")\npython -m spacy download de_core_news_sm")
    elif spacy_text.strip():
        try:
            doc = nlp(spacy_text)
            rows = [{"Text": t.text, "Lemma": t.lemma_, "POS": t.pos_} for t in doc]
            df_spacy = pd.DataFrame(rows, dtype=str)
            try:
                st.dataframe(df_spacy.head(MAX_SPACY_DISPLAY_ROWS), use_container_width=True, hide_index=True)
            except Exception:
                st.table(df_spacy.head(MAX_SPACY_DISPLAY_ROWS))
            adjs = sorted({t.lemma_.lower() for t in doc if t.pos_ == "ADJ" and not t.is_stop})
            st.caption("Adjektive (Sound-Beschreibung): " + (", ".join(adjs) if adjs else "—"))
        except Exception:
            st.error("Fehler beim Rendern der spaCy-Analyse.")

    st.subheader("WAV-Analyse")
    uploaded_wav = st.file_uploader("WAV-Datei hochladen", type=["wav"])
    if uploaded_wav is not None:
        try:
            audio_result = analyze_wav_bytes(uploaded_wav.read(), filename=uploaded_wav.name)
            metrics = audio_result.to_record()
            st.metric("Peak dBFS", metrics["peak_dbfs"])
            st.metric("RMS dBFS", metrics["rms_dbfs"])
            st.metric("Dauer", f"{metrics['duration_seconds']} s")
            st.write(
                {
                    "sample_rate": metrics["sample_rate"],
                    "channels": metrics["channels"],
                    "crest_factor_db": metrics["crest_factor_db"],
                    "dominant_frequency_hz": metrics["dominant_frequency_hz"],
                    "clipping_samples": metrics["clipping_samples"],
                }
            )
            for note in metrics["notes"]:
                st.caption(note)
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("Persönliche Presets")
    user_id = st.text_input("User-ID", value=st.session_state["session_id"])
    with st.form("personal_preset_form"):
        preset_name = st.text_input("Preset-Name", placeholder="Kick Cleanup")
        preset_plugin = st.text_input("Plugin", placeholder="FabFilter Pro-Q 3")
        preset_category = st.selectbox("Kategorie", ["eq", "dynamics", "saturation", "reverb", "custom"], index=0)
        preset_description = st.text_area("Beschreibung", height=70)
        preset_tags = st.text_input("Tags", placeholder="kick, low-end, cleanup")
        save_preset = st.form_submit_button("Preset speichern")
        if save_preset:
            try:
                preset = PERSONAL_PRESETS.add_preset(
                    PersonalPreset(
                        user_id=user_id,
                        name=preset_name,
                        plugin=preset_plugin,
                        category=preset_category,
                        description=preset_description,
                        tags=tuple(tag.strip() for tag in preset_tags.split(",") if tag.strip()),
                    )
                )
                st.success(f"Preset gespeichert: {preset.preset_id}")
            except ValueError as exc:
                st.error(str(exc))

    user_presets = PERSONAL_PRESETS.list_presets(user_id)
    if user_presets:
        preset_df = pd.DataFrame(
            [
                {
                    "Name": preset.name,
                    "Plugin": preset.plugin,
                    "Kategorie": preset.category,
                    "Tags": ", ".join(preset.tags),
                }
                for preset in user_presets
            ]
        )
        st.dataframe(preset_df, use_container_width=True)
    else:
        st.caption("Noch keine persönlichen Presets gespeichert.")
