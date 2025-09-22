import json, os
from pathlib import Path

def main():
    import yake  # nur lokal nötig
    data_path = Path("data/sample_corpus.json")
    out_path = Path("data/keyphrases.json")
    texts = json.loads(data_path.read_text(encoding="utf-8"))

    kw = yake.KeywordExtractor(lan="de", n=3, top=8)  # n: bis zu 3er-Phrasen
    rows = []
    for i, t in enumerate(texts):
        phrases = [k for k, _ in kw.extract_keywords(t)]
        rows.append({"post_id": i, "phrases": phrases})

    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ wrote {out_path} ({len(rows)} items)")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
    # build_keyphrases.py
    # Created by SM on 07.09.25.
