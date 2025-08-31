#!/usr/bin/env bash
set -euo pipefail

# Resolve project root
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
cd "$ROOT"

# --- Environment hygiene ---
# Quieter Torch & tokenizers; no telemetry; optional warnings off
export PYTORCH_JIT=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONWARNINGS=ignore
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_SERVER_ENABLE_FILE_WATCHER=false
# --- venv activation (fail clearly if missing) ---
if [[ ! -f "$ROOT/venv/bin/activate" ]]; then
  echo "[ERROR] Python venv not found at $ROOT/venv. Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi
source "$ROOT/venv/bin/activate"

# --- Ensure dependencies are installed ---
if [[ -f "$ROOT/requirements.txt" ]]; then
  echo "[INFO] Installing dependencies from requirements.txt..."
  pip install -r "$ROOT/requirements.txt"
fi

# --- Run Streamlit ---
PORT="${PORT:-8080}"
echo "[INFO] Starting Streamlit on port $PORT (headless)"
$PY -m streamlit run app/app.py --server.port "$PORT" --server.headless true --server.fileWatcherType none
