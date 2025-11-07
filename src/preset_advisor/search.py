"""
CLI entry or importable API for preset_advisor
Usage:
    python -m preset_advisor.search "Kick klingt dumpf"
"""

import sys, json
from preset_advisor.core import run_search

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m preset_advisor.search '<query>' [backend]")
        sys.exit(1)
    query = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "sbert"
    res = run_search(query, backend=backend, top_k=5)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
