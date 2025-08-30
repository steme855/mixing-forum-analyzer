#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
cd "$ROOT"
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download de_core_news_sm || true
echo "Setup done."
