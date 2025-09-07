import json, os, re
from pathlib import Path

ADJ_TERMS = ["boxig","mulmig","harsch","scharf","nasal","weich","knackig","dull","punchy","hart","trocken"]
DYN_HINTS = r"\b(sidechain|parallel|kompressor|ratio|threshold|attack|release)\b"
SIBILANCE = r"\b(s-laute|s-laut|zischelig|sibilanz|sibilanzen|de-esser)\b"
FREQ = r"\b\d{2,4}\s*(?:k?hz)\b"  # 50 Hz – 8 kHz Bereich

def tag_post(nlp, text: str):
    tags = set()
    t = text.lower()
    # adj terms (via PhraseMatcher wäre feiner, hier simpel)
    if any(term in t for term in ADJ_TERMS):
        tags.add("tone_adj")
    if re.search(DYN_HINTS, t):
        tags.add("dynamics")
    if re.search(SIBILANCE, t):
        tags.add("sibilance")
    if re.search(FREQ, t):
        tags.add("freq")
    # snare/kick/bass heuristics
    if "snare" in t: tags.add("snare")
    if "kick" in t or "kickdrum" in t or "bassdrum" in t: tags.add("kick")
    if "bass" in t: tags.add("bass")
    return sorted(tags)

def main():
    import spacy
    try:
        nlp = spacy.load("de_core_news_sm")
    except Exception:
        import spacy.cli as spacy_cli
        spacy_cli.download("de_core_news_sm")
        nlp = spacy.load("de_core_news_sm")

    texts = json.loads(Path("data/sample_corpus.json").read_text(encoding="utf-8"))
    out = []
    for i, txt in enumerate(texts):
        out.append({"post_id": i, "tags": tag_post(nlp, txt)})

    Path("data/rule_tags.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ wrote data/rule_tags.json")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
