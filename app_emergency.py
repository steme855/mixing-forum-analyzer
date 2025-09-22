#!/usr/bin/env python3
"""
Mixing Forum Analyzer - EMERGENCY VERSION
Ohne spaCy und sentence-transformers (die blockieren)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
import json
from pathlib import Path
import re

# Streamlit Config
st.set_page_config(
    page_title="Mixing Forum Analyzer",
    page_icon="🎚",
    layout="wide"
)

st.title("🎚 Mixing Forum Analyzer")
st.markdown("**KI-gestützte Suche nach ähnlichen Audio-Engineering Problemen**")

# Status
st.sidebar.error("⚠️  NOTFALL-MODUS")
st.sidebar.info("spaCy und SBERT deaktiviert")
st.sidebar.success("✅ TF-IDF funktioniert")

# Korpus laden (ohne externe Dependencies)
@st.cache_data
def load_simple_corpus():
    """Lädt Korpus ohne externe Dependencies"""

    corpus_path = Path("data/sample_corpus.json")

    if corpus_path.exists():
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            st.warning(f"Korpus-Fehler: {e}")

    # Fallback-Korpus
    fallback = [
        "Snare klingt zu harsch und metallisch im Mix",
        "Kick Drum hat zu wenig Punch und verschwindet im Bass",
        "Bass maskiert die Kick und macht den Mix muddy und dumpf",
        "Vocals sitzen nicht im Mix und klingen dünn und schwach",
        "Hi-Hats sind zu schrill und stechen durch den ganzen Mix",
        "Der gesamte Mix klingt leblos und überkomprimiert",
        "Gitarren kämpfen um den gleichen Frequenzbereich und maskieren sich",
        "Der Mix ist zu bass-lastig und dröhnt in kleinen Räumen",
        "Die Stereobreite fehlt komplett, alles klingt mono",
        "Reverb klingt künstlich wie Badezimmer-Hall",
        "Kompression zerstört die Dynamik und macht alles platt",
        "EQ klingt harsch, besonders im Hochtonbereich",
        "Deessing greift zu stark ein und macht Vocals dumpf",
        "Sidechain-Kompression pumpt zu extrem",
        "Monitoring ist schlecht kalibriert und täuscht über den Mix",
        "Low-End ist nicht tight und verschmiert",
        "Midrange ist zu prominent und ermüdend",
        "Transienten gehen durch Overprocessing verloren",
        "Panning ist unausgewogen und einseitig",
        "Mastering-Chain ist zu aggressiv eingestellt"
    ]

    st.info(f"📦 Fallback-Korpus mit {len(fallback)} Problemen geladen")
    return fallback

# Einfache TF-IDF Suche (ohne ML-Dependencies)
@st.cache_data
def build_simple_search_index(corpus_texts):
    """Baut TF-IDF Index ohne externe ML-Libs"""

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words=None  # Keine deutschen Stopwords ohne spaCy
        )

        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
        return vectorizer, tfidf_matrix

    except Exception as e:
        st.error(f"❌ TF-IDF Fehler: {e}")
        return None, None

def search_problems(query, corpus, vectorizer, tfidf_matrix, max_results=5):
    """Sucht ähnliche Probleme mit TF-IDF"""

    if not query.strip():
        return []

    try:
        # Query-Vektor erstellen
        query_vec = vectorizer.transform([query.strip()])

        # Ähnlichkeiten berechnen
        similarities = (tfidf_matrix * query_vec.T).toarray().flatten()

        # Top-Ergebnisse
        top_indices = np.argsort(similarities)[::-1][:max_results]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.01:  # Minimum threshold
                results.append({
                    'text': corpus[idx],
                    'score': score,
                    'index': idx
                })

        return results

    except Exception as e:
        st.error(f"❌ Suchfehler: {e}")
        return []

# Basis-Presets ohne externe Dependencies
def get_simple_presets(problem_text):
    """Einfache Preset-Vorschläge basierend auf Keywords"""

    text_lower = problem_text.lower()
    suggestions = []

    preset_map = {
        "snare": {
            "frequencies": "200 Hz (Körper), 5 kHz (Präsenz)",
            "techniques": "EQ-Boost, Kompression mit mittlerer Ratio",
            "tip": "Vermeide harsche Frequenzen um 3-4 kHz"
        },
        "kick": {
            "frequencies": "60-80 Hz (Sub), 2-5 kHz (Click)",
            "techniques": "Sub-Boost, Hochpass für Klarheit",
            "tip": "Sidechain-Kompression gegen Bass"
        },
        "vocal": {
            "frequencies": "800 Hz (Wärme), 3-5 kHz (Präsenz)",
            "techniques": "Presence-Boost, De-Essing",
            "tip": "Kompression für gleichmäßige Lautstärke"
        },
        "bass": {
            "frequencies": "40-80 Hz (Fundament), 800 Hz (Definition)",
            "techniques": "Hochpass um 30 Hz, Kompression",
            "tip": "Mono unter 100 Hz für Tightness"
        },
        "gitarre": {
            "frequencies": "100 Hz (Hochpass), 3 kHz (Präsenz)",
            "techniques": "Midrange-Sculpting",
            "tip": "Panning für Platz im Mix"
        }
    }

    for instrument, preset in preset_map.items():
        if instrument in text_lower:
            suggestions.append(f"**{instrument.title()}**: {preset['frequencies']}")
            suggestions.append(f"• {preset['techniques']}")
            suggestions.append(f"• {preset['tip']}")

    return suggestions

# Main App
def main():
    # Korpus laden
    with st.spinner("Lade Korpus..."):
        corpus = load_simple_corpus()

    if not corpus:
        st.error("❌ Konnte Korpus nicht laden")
        return

    # Search Index bauen
    with st.spinner("Baue Suchindex..."):
        vectorizer, tfidf_matrix = build_simple_search_index(corpus)

    if vectorizer is None:
        st.error("❌ Konnte Suchindex nicht erstellen")
        return

    st.success(f"✅ {len(corpus)} Mixing-Probleme indexiert")

    # Search Interface
    st.subheader("🔍 Problem-Suche")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Beschreibe dein Mixing-Problem:",
            placeholder="z.B. Snare klingt harsch und metallisch...",
            height=100
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        max_results = st.slider("Max. Ergebnisse:", 3, 10, 5)
        search_clicked = st.button("🔍 Suchen", type="primary")

    # Suche ausführen
    if search_clicked and query.strip():
        with st.spinner("Suche läuft..."):
            results = search_problems(query, corpus, vectorizer, tfidf_matrix, max_results)

        if results:
            st.subheader(f"🎯 Gefunden: {len(results)} ähnliche Probleme")

            for i, result in enumerate(results):
                with st.expander(f"#{i+1} • {result['text'][:60]}... (Score: {result['score']:.3f})"):
                    st.write(result['text'])

                    # Einfache Preset-Vorschläge
                    suggestions = get_simple_presets(result['text'])
                    if suggestions:
                        st.markdown("**💡 Lösungsansätze:**")
                        for suggestion in suggestions:
                            st.markdown(f"• {suggestion}")
        else:
            st.info("Keine relevanten Ergebnisse gefunden. Versuche andere Keywords.")

    elif search_clicked:
        st.warning("Bitte gib eine Suchanfrage ein.")

    # Info
    st.markdown("---")
    st.markdown("⚠️  **Notfall-Modus**: Diese Version läuft ohne spaCy/SBERT wegen Import-Problemen")
    st.markdown("🔧 **Funktionen**: TF-IDF Suche, Basis-Presets, Keyword-Matching")

if __name__ == "__main__":
    main()
