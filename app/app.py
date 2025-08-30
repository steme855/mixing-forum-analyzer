import json
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import spacy
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Mixing Forum Analyzer", layout="wide")

st.title("🎛️ Mixing Forum Analyzer — Woche 1 Demo")
st.write("Semantische Suche mit Sentence-Embeddings auf Beispiel-Posts. Bonus: spaCy POS/Adjektive.")

# ------------------------------
# Data loading
# ------------------------------
@st.cache_data
def load_corpus() -> List[str]:
    with open("data/sample_corpus.json", "r", encoding="utf-8") as f:
        return json.load(f)

corpus = load_corpus()

# ------------------------------
# Embedding model
# ------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=True)
def encode_corpus(texts: List[str]) -> np.ndarray:
    model = load_model()
    emb = model.encode(texts, normalize_embeddings=True)
    return emb

corp_emb = encode_corpus(corpus)

# ------------------------------
# Preset suggestions (robust JSON + fuzzy matching + structured schema)
# ------------------------------
@st.cache_data
def load_preset_map() -> Dict[str, Any]:
    """Load preset map from data/ or project root. Return {} if missing/empty/invalid."""
    paths = ["data/preset_map.json", "preset_map.json"]
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    continue
                return json.loads(content)
        except FileNotFoundError:
            continue
        except Exception:
            # JSON decode or other errors -> try next path
            continue
    return {}

preset_map = load_preset_map()

# Helpers for structured schema (schema_version + entries)
def is_structured_schema(pm: Dict[str, Any]) -> bool:
    return isinstance(pm, dict) and "entries" in pm and isinstance(pm["entries"], list)

def render_action(a: Dict[str, Any]) -> str:
    t = a.get("type", "step")
    if t == "eq":
        band = a.get("band", "bell")
        freq = a.get("freq_hz", "?")
        gain = a.get("gain_db", 0)
        q = a.get("q", 1.0)
        return f"EQ: {band} @ {freq} Hz, {gain} dB, Q {q}"
    if t == "de-esser":
        pos = a.get("position", "")
        red = a.get("reduction_db", 2)
        extra = f" ({pos.replace('_',' ')})" if pos else ""
        return f"De-Esser{extra}: {red} dB"
    if t == "dynamic_eq":
        freq = a.get("freq_hz", "?")
        thr = a.get("threshold_db", "?")
        ratio = a.get("ratio", "?")
        return f"Dyn-EQ @ {freq} Hz, Th {thr} dB, Ratio {ratio}"
    if t == "transient":
        mode = a.get("mode", "attack_plus")
        amt = a.get("amount", 0.3)
        return f"Transient ({mode}): {amt}"
    if t == "parallel_comp":
        ratio = a.get("ratio", 4.0)
        blend = a.get("blend", 0.3)
        return f"Parallel Comp: Ratio {ratio}:1, Blend {int(blend*100)}%"
    if t == "clipper":
        amt = a.get("amount_db", 1.0)
        return f"Clipper: {amt} dB"
    if t == "sidechain":
        duck = a.get("duck_db", 2.0)
        atk = a.get("attack_ms", 10)
        rel = a.get("release_ms", 120)
        return f"Sidechain: Duck {duck} dB, Attack {atk} ms, Release {rel} ms"
    if t == "hp_filter_sc":
        freq = a.get("freq_hz", 120)
        return f"SC-HPF @ {freq} Hz"
    # Fallback text
    return ", ".join([f"{k}={v}" for k, v in a.items()])

# Fuzzy + synonym-aware preset matcher (flat map)
def suggest_presets_flat(text: str, pm: Dict[str, Any]) -> list:
    if not text or not pm:
        return []
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
        "maskiert": ["maskiert", "verdeckt", "überlagert"]
    }
    for syns in SYNS.values():
        if any(s in q for s in syns):
            q += " " + " ".join(syns)

    candidates = list(pm.keys())
    if not candidates:
        return []

    scored = []
    for key in candidates:
        k = key.lower()
        s1 = fuzz.token_set_ratio(q, k)
        s2 = fuzz.partial_ratio(q, k)
        score = max(s1, s2)
        if score >= 70:
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

# Structured schema matcher (tiered actions)
def suggest_presets_structured(text: str, pm: Dict[str, Any], level: str) -> list:
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

# Single entry point (uses structured schema if available)
def suggest_presets(text: str, level: str) -> list:
    if is_structured_schema(preset_map):
        return suggest_presets_structured(text, preset_map, level)
    # fallback flat map
    return suggest_presets_flat(text, preset_map)

# ------------------------------
# UI
# ------------------------------
query = st.text_area(
    "🔎 Beschreibe dein Mixing-Problem (z. B. 'Snare zu boxig, mehr Punch auf Kick')",
    height=100,
)

# Preset-Stufe (nur sichtbar, wenn eine Map vorhanden ist)
preset_level = st.selectbox(
    "Preset-Intensität",
    options=["konservativ", "standard", "aggressiv"],
    index=1,
    help="Steuert die vorgeschlagenen Maßnahmen bei strukturiertem Mapping. Bei einfachem Mapping ohne Stufen hat dies keinen Effekt."
)

col1, col2 = st.columns(2)

# ------------------------------
# Left: Semantic search + presets + explanations
# ------------------------------
with col1:
    st.subheader("Semantische Suche")
    search_clicked = st.button("Ähnliche Posts finden")

    # Domain lexicon for lightweight query expansion
    LEX = {
        "snare": ["snare", "rimshot"],
        "kick": ["kick", "kickdrum", "bassdrum"],
        "laut": ["laut", "zu laut", "pegel hoch", "clipping", "übersteuern"],
        "boxig": ["boxig", "mulmig", "mittig", "muddy"],
        "scharf": ["scharf", "harsch", "zischelig"],
        "punch": ["punch", "attack", "anschlag"],
        "vocal": ["vocal", "gesang", "vox"],
        "bass": ["bass", "subbass", "sub-bass"],
    }

    def expand_query(q: str) -> str:
        ql = q.lower()
        extra = []
        for head, syns in LEX.items():
            if any(tok in ql for tok in syns + [head]):
                extra += syns + [head]
        if extra:
            return q + " " + " ".join(sorted(set(extra)))
        return q

    INST_WORDS = {"snare": 1.15, "kick": 1.10, "vocal": 1.05, "bass": 1.05}

    def boost_score(base: float, text_l: str) -> float:
        b = base
        for w, mult in INST_WORDS.items():
            if w in text_l:
                b *= mult
        # boost if numbers with Hz/kHz are present
        if re.search(r"\b\d{2,4}\s*(k?hz)\b", text_l):
            b *= 1.05
        return float(b)

    df = None
    top_idx = []

    if search_clicked:
        if not query.strip():
            st.warning("Bitte eine Query eingeben.")
        else:
            q_expanded = expand_query(query)
            model = load_model()
            qv = model.encode([q_expanded], normalize_embeddings=True)
            sims = (qv @ corp_emb.T).ravel()

            # boosting + simple re-ranking
            scored = []
            for i, s in enumerate(sims):
                txt = corpus[i].lower()
                scored.append((boost_score(float(s), txt), i))
            scored.sort(reverse=True)
            top_idx = [i for _, i in scored[:5]]

            df = pd.DataFrame({
                "Ähnlichkeit": [round(scored[k][0], 3) for k in range(min(5, len(scored)))],
                "Post": [corpus[i] for i in top_idx]
            })
            st.subheader("Top 5 ähnliche Posts (semantisch)")
            st.dataframe(df, use_container_width=True)

            # Erklärung: Lemma-Overlap
            try:
                nlp = spacy.load("de_core_news_sm")
                q_lem = {t.lemma_.lower() for t in nlp(query) if not (t.is_stop or t.is_punct or t.like_num)}
                st.caption("Warum diese Treffer? (Lemma-Overlap)")
                for rank, i in enumerate(top_idx, 1):
                    d_lem = {t.lemma_.lower() for t in nlp(corpus[i]) if not (t.is_stop or t.is_punct or t.like_num)}
                    overlap = ", ".join(sorted(q_lem & d_lem)) or "—"
                    st.markdown(f"**#{rank}** Overlap: _{overlap}_")
            except Exception:
                st.caption("Erklär-Overlay benötigt spaCy DE.")

            # Preset-Vorschläge (structured bevorzugt)
            recs = suggest_presets(query, level={
                "konservativ": "conservative",
                "standard": "standard",
                "aggressiv": "aggressive"
            }[preset_level])
            st.subheader("🔧 Vorschläge (Preset/Chain)")
            if recs:
                for r in recs:
                    st.markdown(f"- {r}")
            else:
                st.caption("Keine direkten Vorschläge – Formulierung verfeinern (z. B. 'snare boxig', 'bass maskiert kick').")

            # Optionaler Export-Button (sicher: nur wenn df existiert)
            if st.button("Ergebnisse als CSV speichern"):
                try:
                    df.to_csv("data/last_query_results.csv", index=False, encoding="utf-8")
                    st.success("CSV gespeichert: data/last_query_results.csv")
                except Exception as e:
                    st.error(f"CSV-Export fehlgeschlagen: {e}")

# ------------------------------
# Right: spaCy inspection panel
# ------------------------------
with col2:
    st.subheader("spaCy Analyse (DE) — Adjektive & Lemmata")

    @st.cache_resource
    def load_spacy():
        try:
            return spacy.load("de_core_news_sm")
        except Exception:
            st.info("spaCy-Modell fehlt: `python -m spacy download de_core_news_sm`")
            return None

    nlp = load_spacy()
    text_for_spacy = st.text_area(
        "Text für spaCy (optional)",
        value="Die Snare klingt trocken und etwas hart, vielleicht mehr Raumanteil.",
        height=100,
    )

    if nlp is not None:
        doc = nlp(text_for_spacy)
        rows = [{"Text": t.text, "Lemma": t.lemma_, "POS": t.pos_} for t in doc]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        adjectives = [t.text for t in doc if t.pos_ == "ADJ"]
        st.markdown("**Adjektive (Sound-Beschreibung):** " + (", ".join(adjectives) if adjectives else "—"))

