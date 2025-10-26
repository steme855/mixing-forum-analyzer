#!/usr/bin/env python3
import os, sys, runpy
# Simple launcher that ensures src on sys.path and runs the Streamlit app entry
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# Delegate to Streamlit if invoked via `streamlit run app.py`
# or run as normal python (for testing smoke imports)
if __name__ == "__main__":
    runpy.run_path(os.path.join(SRC, "app", "main.py"))
