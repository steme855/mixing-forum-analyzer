def test_imports():
    import numpy, pandas, sklearn, streamlit, spacy
    nlp = spacy.load("de_core_news_sm")
    assert nlp is not None

def test_repo_layout_exists():
    import os
    assert os.path.exists("app/app.py")
    assert os.path.exists("requirements.txt")
