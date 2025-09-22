#!/usr/bin/env python3
"""
Schneller Streamlit Test ohne problematische Imports
"""

import streamlit as st
import sys

st.set_page_config(page_title="Python Check", layout="wide")

st.title("🐍 Python & Streamlit Compatibility Check")

st.subheader("🔍 System Info")
st.write(f"**Python Version:** {sys.version}")
st.write(f"**Python Executable:** {sys.executable}")

py_version = sys.version_info

if py_version.major == 3 and py_version.minor == 12:
    st.error("🔴 Python 3.12 erkannt - das ist die Ursache deiner Probleme!")
    st.markdown("""
    **Probleme mit Python 3.12:**
    - spaCy hängt beim Import
    - sentence-transformers blockiert
    - transformers sehr langsam

    **Lösung:**
    ```bash
    brew install python@3.11
    python3.11 -m venv venv_311
    source venv_311/bin/activate
    pip install -r requirements.txt
    ```
    """)
elif py_version.major == 3 and py_version.minor == 11:
    st.success("✅ Python 3.11 - perfekt für ML!")
else:
    st.info(f"Python {py_version.major}.{py_version.minor} erkannt")

st.subheader("🧪 Import Tests")

# Test nur die funktionierenden Imports
imports_to_test = [
    ("streamlit", "✅ Funktioniert"),
    ("pandas", "✅ Sollte funktionieren"),
    ("numpy", "✅ Sollte funktionieren"),
    ("sklearn", "✅ Sollte funktionieren"),
]

for module, status in imports_to_test:
    try:
        __import__(module)
        st.success(f"{module}: {status}")
    except ImportError:
        st.error(f"{module}: ❌ Import fehlt")

# Problematische Imports warnen
st.subheader("⚠️ Problematische Imports (Python 3.12)")
problematic = [
    "spaCy: Hängt beim Import",
    "sentence-transformers: Deadlock",
    "transformers: Sehr langsam"
]

for problem in problematic:
    st.warning(f"❌ {problem}")

st.markdown("---")
st.info("💡 Nutze app_emergency.py für sofortige Funktionalität ohne problematische Libraries!")

if st.button("🚀 Emergency App starten"):
    st.code("streamlit run app_emergency.py --server.port 8503", language="bash")
