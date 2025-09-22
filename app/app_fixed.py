#!/usr/bin/env python3
"""
Mixing Forum Analyzer - FIXED VERSION (PyTorch/SBERT disabled)
Alle problematischen ML-Models temporär deaktiviert
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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import importlib.util
from collections import defaultdict

# =============================================================================
# EMERGENCY FIX - Deaktiviere problematische Features
# =============================================================================

print("🚀 Starting Mixing Forum Analyzer (FIXED VERSION)")
print("PyTorch/SBERT/spaCy temporarily disabled")

# Environment settings (keep the working ones)
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")  
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_FILE_WATCHER", "false")
# os.environ.setdefault("PYTORCH_JIT", "0")  # DISABLED
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # DISABLED

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

def _quiet_torch_logs():
    """DISABLED - would setup torch logging"""
    print("Torch logging setup disabled")

def _warmup_torch_silently():
    """DISABLED - PyTorch warmup deaktiviert"""
    print("⚠️  PyTorch warmup disabled for stability")
    return None

def load_corpus() -> List[str]:
    """Load mixing problem corpus - basic version"""
    try:
        corpus_path = Path("data/sample_corpus.json")
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"✅ Loaded {len(data)} corpus items")
                    return data
                else:
                    print("⚠️  Unexpected corpus format, using fallback")
    except Exception as e:
        print(f"⚠️  Corpus loading error: {e}")

    # Fallback corpus
    fallback = [
        "Snare klingt zu harsch und metallisch im Mix",
        "Kick Drum hat zu wenig Punch und verschwindet",
        "Bass maskiert die Kick und macht den Mix muddy",
        "Vocals sitzen nicht im Mix und klingen dünn",
        "Hi-Hats zu schrill und dominant",
        "Der gesamte Mix klingt leblos und komprimiert",
        "Gitarren kämpfen um den gleichen Frequenzbereich",
        "Der Mix ist zu bass-lastig und dröhnt",
        "Die Stereobreite fehlt komplett",
        "Alles klingt nach Badezimmer-Reverb"
    ]
    print(f"📦 Using fallback corpus with {len(fallback)} items")
    return fallback

@st.cache_data
def load_preset_map():
    """Load preset mappings"""
    try:
        preset_path = Path("data/preset_map.json")
        if preset_path.exists():
            with open(preset_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️  Preset map error: {e}")

    # Fallback presets
    return {
        "snare": {
            "freq_ranges": ["200 Hz", "5 kHz"],
            "techniques": ["EQ boost", "Compression"],
            "description": "Snare punch and presence"
        },
        "kick": {
            "freq_ranges": ["60-80 Hz", "2-5 kHz"],
            "techniques": ["Sub boost", "Click enhancement"],
            "description": "Kick power and definition"
        },
        "vocal": {
            "freq_ranges": ["800 Hz", "3-5 kHz"],
            "techniques": ["Presence boost", "De-essing"],
            "description": "Vocal clarity and warmth"
        }
    }

def build_tfidf(corpus_texts: List[str]):
    """Build TF-IDF vectorizer - WORKING VERSION"""
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=None
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
        print(f"✅ TF-IDF built: {tfidf_matrix.shape}")
        return vectorizer, tfidf_matrix
    except Exception as e:
        print(f"❌ TF-IDF error: {e}")
        return None, None

def load_sbert(model_name: str):
    """DISABLED - SBERT loading deaktiviert"""
    print("⚠️  SBERT disabled - using TF-IDF only")
    return None

def embed_corpus(model_name: str, docs: List[str]):
    """DISABLED - Corpus embedding deaktiviert"""  
    print("⚠️  Corpus embedding disabled")
    return None

def load_spacy():
    """DISABLED - spaCy loading deaktiviert"""
    print("⚠️  spaCy disabled for stability")
    return None

# =============================================================================
# MAIN STREAMLIT APP - BASIC VERSION
# =============================================================================

def main():
    """Main Streamlit application"""

    st.set_page_config(
        page_title="Mixing Forum Analyzer",
        page_icon="🎚",
        layout="wide"
    )

    st.title("🎚 Mixing Forum Analyzer")
    st.markdown("**KI-gestützte Suche nach ähnlichen Audio-Engineering Problemen**")

    # Status info
    st.sidebar.success("✅ TF-IDF Mode (Stable)")
    st.sidebar.warning("⚠️  SBERT/PyTorch disabled")
    st.sidebar.info("💡 This is a stability-focused version")

    # Load data (basic version)
    with st.spinner("Loading corpus..."):
        corpus = load_corpus()
        preset_map = load_preset_map()

    # Build search engine
    with st.spinner("Building search index..."):
        vectorizer, tfidf_matrix = build_tfidf(corpus)

    if vectorizer is None:
        st.error("❌ Could not build search index")
        return

    st.success(f"✅ Search ready! {len(corpus)} problems indexed")

    # Search interface
    st.subheader("🔍 Problem Search")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Describe your mixing problem:",
            placeholder="e.g., Snare klingt zu harsch und metallisch...",
            height=100
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        max_results = st.slider("Max results:", 3, 10, 5)
        search_clicked = st.button("🔍 Search", type="primary")

    # Perform search
    if search_clicked and query.strip():
        with st.spinner("Searching..."):
            try:
                # TF-IDF search
                query_vec = vectorizer.transform([query.strip()])
                similarities = (tfidf_matrix * query_vec.T).toarray().flatten()

                # Get top results
                top_indices = np.argsort(similarities)[::-1][:max_results]

                st.subheader(f"🎯 Search Results")

                results_found = False
                for i, idx in enumerate(top_indices):
                    score = similarities[idx]
                    if score > 0.01:  # Minimum relevance threshold
                        results_found = True

                        with st.expander(f"#{i+1} • {corpus[idx][:60]}... (Score: {score:.3f})"):
                            st.write(corpus[idx])

                            # Check for presets
                            problem_text = corpus[idx].lower()
                            suggestions = []

                            for preset_name, preset_data in preset_map.items():
                                if preset_name in problem_text:
                                    suggestions.append(f"**{preset_name.title()}**: {preset_data.get('description', '')}")

                            if suggestions:
                                st.markdown("**💡 Suggested approaches:**")
                                for suggestion in suggestions:
                                    st.markdown(f"• {suggestion}")

                if not results_found:
                    st.info("No highly relevant results found. Try different keywords.")

            except Exception as e:
                st.error(f"❌ Search error: {e}")

    elif search_clicked:
        st.warning("Please enter a search query")

    # Footer
    st.markdown("---")
    st.markdown("🔧 **Fixed Version** - PyTorch/SBERT disabled for stability")
    st.markdown("For full features, resolve PyTorch/sentence-transformers loading issues")

if __name__ == "__main__":
    main()
