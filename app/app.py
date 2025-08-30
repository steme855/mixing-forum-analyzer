import json
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import time
from rapidfuzz import fuzz
import os
os.environ["PYTORCH_JIT"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ.setdefault("DEBUG_STREAMLIT", "0")

# Suppress noisy logs from transformers/SBERT
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
st.set_page_config(page_title="Mixing Forum Analyzer", layout="wide")

st.title("🎛️ Mixing Forum Analyzer — Woche 1 Demo")
st.write("Semantische Suche mit Sentence-Embeddings auf Beispiel-Posts. Bonus: spaCy POS/Adjektive.")

# Header badges
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

# Header status for last search
if "mode_label" in st.session_state and "elapsed_ms" in st.session_state:
    st.caption(f"Modus: {st.session_state['mode_label']} · letzte Suche: {st.session_state['elapsed_ms']:.0f} ms")

# ------------------------------
# Data loading
# ------------------------------
@st.cache_data
def load_corpus() -> List[str]:
    """Load corpus from data/sample_corpus.json. If missing/invalid, return a small fallback list."""
    try:
        with open("data/sample_corpus.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return [
            "Die Snare ist zu laut und harsch",
            "Kick zu weich, es fehlt der Punch",
            "Vocals klingen nasal, 800 Hz absenken",
            "Bass maskiert die Kick, Sidechain nötig",
            "S-Laute sind scharf, De-Esser einsetzen",
        ]

@st.cache_data(show_spinner=False)
def load_preset_map(path="data/preset_map.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                # Leere Datei – failsafe
                return {"schema_version": "1.0", "entries": []}
            return json.loads(txt)
    except Exception as e:
        st.warning(f"Preset-Map konnte nicht geladen werden ({e}). Fallback aktiv.")
        return {"schema_version": "1.0", "entries": []}

corpus = load_corpus()
preset_map = load_preset_map()

# ------------------------------
# Embedding model
# ------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # default

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        st.error(f"SBERT konnte nicht geladen werden (Fallback auf TF‑IDF): {e}")
        return None

@st.cache_data(show_spinner=False)
def embed_corpus(corpus_texts: List[str], model_name: str):
    model = load_model(model_name)
    if model is None:
        return None
    return model.encode(corpus_texts, normalize_embeddings=True)

@st.cache_data(show_spinner=False)
def build_tfidf(corpus_texts: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    X = vect.fit_transform(corpus_texts)
    return vect, X

@st.cache_resource
def load_spacy():
    try:
        import spacy  # lazy import to avoid slow startup
        return spacy.load("de_core_news_sm")
    except Exception:
        st.info("spaCy-Modell fehlt: `python -m spacy download de_core_news_sm`")
        return None

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

#
# Compact tooltips for common preset actions

def tooltip_for(rec_str: str) -> str:
    tmap = {
        "de-esser": "De-Esser: 4–6 dB bei 6–8 kHz",
        "eq:": "EQ: gezielte Bänder, Gain moderat, Q nach Bedarf",
        "dynamic_eq": "Dyn‑EQ: nur bei Überschreitung absenken",
        "dyn-eq": "Dyn‑EQ: nur bei Überschreitung absenken",
        "transient": "Transient: Attack erhöhen für mehr Punch",
        "parallel comp": "Parallel Comp: Blend 20–40%",
        "clipper": "Clipper: sanftes Limiting, 1–2 dB",
        "sidechain": "Sidechain: Kick duckt Bass 2–4 dB",
        "sc-hpf": "SC‑HPF: SC unter 80–150 Hz filtern",
    }
    s = rec_str.lower()
    for k, v in tmap.items():
        if k in s:
            return v
    return "Preset-Hinweis"

# ------------------------------
# UI
# ------------------------------

#
# Initialisieren, falls noch nicht vorhanden
if "query_text" not in st.session_state:
    st.session_state["query_text"] = ""

# Beispiel-Buttons OBERHALB des Textfelds: zuerst State setzen, dann rerun
ex_cols = st.columns(5)
with ex_cols[0]:
    if st.button("Snare zu laut", use_container_width=True):
        st.session_state["query_text"] = "Snare zu laut"
        st.session_state["trigger_search"] = True
        st.toast("Query gesetzt: Snare zu laut")
        st.rerun()
with ex_cols[1]:
    if st.button("Kick weich / Punch", use_container_width=True):
        st.session_state["query_text"] = "Kick zu weich, mehr Punch"
        st.session_state["trigger_search"] = True
        st.toast("Query gesetzt: Kick zu weich, mehr Punch")
        st.rerun()
with ex_cols[2]:
    if st.button("S-Laute scharf", use_container_width=True):
        st.session_state["query_text"] = "S-Laute zu scharf, De-Esser?"
        st.session_state["trigger_search"] = True
        st.toast("Query gesetzt: S-Laute zu scharf, De-Esser?")
        st.rerun()
with ex_cols[3]:
    if st.button("Bass maskiert Kick", use_container_width=True):
        st.session_state["query_text"] = "Bass maskiert Kick, Sidechain einrichten"
        st.session_state["trigger_search"] = True
        st.toast("Query gesetzt: Bass maskiert Kick, Sidechain einrichten")
        st.rerun()
with ex_cols[4]:
    if st.button("✖︎ Clear", use_container_width=True):
        st.session_state["query_text"] = ""
        st.session_state["trigger_search"] = False
        st.toast("Query geleert")
        st.rerun()

# Eingabe: optional einzeilig (Enter = senden) oder mehrzeilig

def _trigger_search_cb():
    st.session_state["trigger_search"] = True

label_txt = "🔎 Beschreibe dein Mixing-Problem (z. B. 'Snare zu boxig, mehr Punch auf Kick')"
placeholder_txt = st.session_state.get("query_text", "") or "z. B. 'Snare zu boxig, mehr Punch auf Kick'"

if st.session_state.get("send_on_enter", False):
    query = st.text_input(
        label_txt,
        key="query_text",
        placeholder=placeholder_txt,
        on_change=_trigger_search_cb,
    )
else:
    query = st.text_area(
        label_txt,
        key="query_text",
        placeholder=placeholder_txt,
        height=100,
    )

# Hotkey-Hinweis (Workaround)
if st.session_state.get("send_on_enter", False):
    st.caption("↩️ **Enter** startet sofort die Suche (einzeiliges Feld).")
else:
    st.caption("⌨️ Tipp: **Ctrl+Enter** / **⌘+Enter** – dann sofort auf *Ähnliche Posts finden* klicken.")

# Preset-Stufe (nur sichtbar, wenn eine Map vorhanden ist)
preset_level = st.selectbox(
    "Preset-Intensität",
    options=["konservativ", "standard", "aggressiv"],
    index=1,
    help="Steuert die vorgeschlagenen Maßnahmen bei strukturiertem Mapping. Bei einfachem Mapping ohne Stufen hat dies keinen Effekt."
)
st.caption("ℹ️ **Preset-Intensität**: *konservativ* = dezente Eingriffe, *standard* = ausgewogen, *aggressiv* = deutlichere Eingriffe. Wirkt nur bei strukturiertem Mapping.")


# Checkbox für Enter-to-send
send_on_enter = st.sidebar.checkbox(
    "Senden mit Enter (einzeilig)",
    value=False,
    help="Aktiviere dies, um das Eingabefeld auf einzeilig zu stellen. Enter löst dann direkt die Suche aus.",
)
st.session_state["send_on_enter"] = send_on_enter

use_semantics = st.sidebar.toggle(
    "Semantische Embeddings (SBERT)",
    value=False,
    help="Aktivieren lädt ein SBERT-Modell (~80MB). Wenn aus, wird TF‑IDF genutzt."
)

# Modell-Auswahl im Sidebar bei aktiviertem SBERT
model_options = {
    "Multilingual MiniLM (kompakt, 384d)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Multilingual DistilUSE (größer, 512d)": "sentence-transformers/distiluse-base-multilingual-cased-v2",
}
if use_semantics:
    sel = st.sidebar.selectbox(
        "SBERT-Modell",
        options=list(model_options.keys()),
        index=0,
        help="Wähle das Embedding-Modell. MiniLM ist schneller/kleiner; DistilUSE ist größer.",
    )
    st.session_state["sbert_model_name"] = model_options[sel]
else:
    st.session_state["sbert_model_name"] = MODEL_NAME

k = st.sidebar.slider("Anzahl Treffer", min_value=3, max_value=10, value=5)
show_explanations = st.sidebar.checkbox("Erklärungen anzeigen (Lemma-Overlap)", value=True)
show_overlap_chart = st.sidebar.checkbox("Mini-Bar-Chart anzeigen", value=True)

if use_semantics:
    ready = st.session_state.get("sbert_ready", False)
    st.sidebar.caption(f"SBERT-Status: {'✅ geladen' if ready else '⏳ lädt bei erster Suche'}")

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

    if search_clicked or st.session_state.get("trigger_search"):
        if not query.strip():
            st.warning("Bitte eine Query eingeben.")
        else:
            q_expanded = expand_query(query)
            if use_semantics:
                model_name = st.session_state.get("sbert_model_name", MODEL_NAME)
                t0 = time.perf_counter()
                with st.spinner("Lade SBERT… (einmalig)"):
                    model = load_model(model_name)
                    corp_emb = embed_corpus(corpus, model_name)
                    st.session_state["sbert_ready"] = bool(model is not None and corp_emb is not None)
                if model is not None and corp_emb is not None:
                    qv = model.encode([q_expanded], normalize_embeddings=True)
                    sims = (qv @ corp_emb.T).ravel()
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    mode_label = "SBERT (semantisch)"
                else:
                    # graceful fallback
                    t0 = time.perf_counter()
                    vect, X = build_tfidf(corpus)
                    q_vec = vect.transform([q_expanded])
                    sims = (q_vec @ X.T).toarray().ravel()
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    mode_label = "TF‑IDF (lexikalisch)"
            else:
                t0 = time.perf_counter()
                vect, X = build_tfidf(corpus)
                q_vec = vect.transform([q_expanded])
                sims = (q_vec @ X.T).toarray().ravel()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                mode_label = "TF‑IDF (lexikalisch)"

            st.session_state["mode_label"] = mode_label
            st.session_state["elapsed_ms"] = elapsed_ms

            # boosting + simple re-ranking
            scored = []
            for i, s in enumerate(sims):
                txt = corpus[i].lower()
                scored.append((boost_score(float(s), txt), i))
            scored.sort(reverse=True)
            k = min(k, len(scored))
            top_idx = [i for _, i in scored[:k]]

            df = pd.DataFrame({
                "Ähnlichkeit": [round(scored[idx][0], 3) for idx in range(k)],
                "Post": [corpus[i] for i in top_idx]
            })
            st.subheader(f"Top {k} ähnliche Posts")
            st.dataframe(df, use_container_width=True)
            st.caption(f"⏱️ {mode_label} · Berechnungszeit: {elapsed_ms:.0f} ms")


            # Direct CSV download (in-memory)
            try:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("CSV downloaden", data=csv_bytes, file_name="last_query_results.csv", mime="text/csv")
            except Exception:
                pass

            # Erklärung: Lemma-Overlap (conditional on sidebar flags)
            nlp = load_spacy()
            if nlp is not None:
                q_lem = {t.lemma_.lower() for t in nlp(query) if not (t.is_stop or t.is_punct or t.like_num)}
                if show_explanations:
                    st.caption("Warum diese Treffer? (Lemma-Overlap)")
                    overlap_counts = []
                    labels = []
                    for rank, i in enumerate(top_idx, 1):
                        d_lem = {t.lemma_.lower() for t in nlp(corpus[i]) if not (t.is_stop or t.is_punct or t.like_num)}
                        ov_set = q_lem & d_lem
                        overlap = ", ".join(sorted(ov_set)) or "—"
                        st.markdown(f"**#{rank}** Overlap: _{overlap}_")
                        overlap_counts.append(len(ov_set))
                        labels.append(f"#{rank}")
                    if show_overlap_chart:
                        try:
                            chart_df = pd.DataFrame({"Treffer": labels, "Overlap": overlap_counts}).set_index("Treffer")
                            if not chart_df.empty and chart_df["Overlap"].sum() > 0:
                                st.bar_chart(chart_df)
                        except Exception:
                            pass
            else:
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
                    tip = tooltip_for(r)
                    st.markdown(f"- <span title=\"{tip}\">{r}</span>", unsafe_allow_html=True)
                st.caption("Bewege den Mauszeiger über einen Vorschlag für kurze Hinweise.")
            else:
                st.caption("Keine direkten Vorschläge – Formulierung verfeinern (z. B. 'snare boxig', 'bass maskiert kick').")

            # Download JSON (inkl. Presets) – NACHDEM "recs" berechnet wurde
            try:
                results_json = {
                    "schema_version": "1.0",
                    "query": query,
                    "expanded_query": q_expanded,
                    "mode": mode_label,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "results": [
                        {"rank": r + 1, "post": corpus[i], "score": round(scored[r][0], 3)}
                        for r, i in enumerate(top_idx)
                    ],
                    "presets": recs or [],
                }
                json_bytes = json.dumps(results_json, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    "JSON downloaden",
                    data=json_bytes,
                    file_name="last_query_results.json",
                    mime="application/json",
                )
            except Exception:
                pass

            # Optionaler Export-Button (sicher: nur wenn df existiert)
            if st.button("Ergebnisse als CSV speichern"):
                try:
                    import os
                    os.makedirs("data", exist_ok=True)
                    df.to_csv("data/last_query_results.csv", index=False, encoding="utf-8")
                    st.success("CSV gespeichert: data/last_query_results.csv")
                except Exception as e:
                    st.error(f"CSV-Export fehlgeschlagen: {e}")

        # Reset trigger_search after search
        st.session_state["trigger_search"] = False

# ------------------------------
# Right: spaCy inspection panel
# ------------------------------
with col2:
    st.subheader("spaCy Analyse (DE) — Adjektive & Lemmata")

    spacy_text = st.text_area(
        "Text für spaCy (optional)",
        value="Die Snare klingt trocken und etwas hart, vielleicht mehr Raumanteil.",
        key="spacy_text",
        height=100,
    )

    nlp = load_spacy()
    if not nlp:
        st.info("spaCy DE nicht verfügbar – installiere mit: `python -m spacy download de_core_news_sm`.")
    elif not spacy_text.strip():
        st.caption("Gib optional Text ein, um Tokens/Lemmata zu sehen.")
    else:
        try:
            doc = nlp(spacy_text)
            rows = [{
                "Text": str(t.text),
                "Lemma": str(t.lemma_),
                "POS": str(t.pos_)
            } for t in doc]
            df_spacy = pd.DataFrame(rows, dtype=str)
            if df_spacy.empty:
                st.caption("Keine Tokens gefunden.")
            else:
                # Robust rendering to avoid rare React errors in st.dataframe
                try:
                    safe_df = df_spacy.reset_index(drop=True).astype(str)
                    st.dataframe(safe_df.head(200), use_container_width=True, hide_index=True)
                except Exception:
                    st.table(df_spacy.head(200))
            adjs = sorted({t.lemma_.lower() for t in doc if t.pos_ == "ADJ" and not t.is_stop})
            st.caption("Adjektive (Sound-Beschreibung): " + (", ".join(adjs) if adjs else "—"))
        except Exception as e:
            st.error("Fehler beim Rendern der spaCy-Analyse.")
            import os as _os
            if _os.getenv("DEBUG_STREAMLIT") == "1":
                st.exception(e)
