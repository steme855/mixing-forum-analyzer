#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download de_core_news_sm
echo "Setup complete. Run: streamlit run app.py"
