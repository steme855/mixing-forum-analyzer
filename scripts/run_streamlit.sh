#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
source "$ROOT/venv/bin/activate"
cd "$ROOT"
exec streamlit run app/app.py --server.port "${PORT:-8080}" --server.headless true
