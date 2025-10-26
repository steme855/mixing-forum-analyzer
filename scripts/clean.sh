#!/usr/bin/env bash
set -euo pipefail
echo "Removing caches and OS cruft..."
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
find . -name ".DS_Store" -type f -delete
echo "Done."
