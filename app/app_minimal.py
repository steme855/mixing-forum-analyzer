import streamlit as st
import pandas as pd

st.set_page_config(page_title="DEBUG", layout="wide")
st.title("🚨 DEBUG - Minimal Test")
st.success("✅ Streamlit lädt!")

st.write("Wenn du das siehst, funktioniert Streamlit grundsätzlich.")

if st.button("Test TF-IDF"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    texts = ["snare laut", "kick weich", "bass maskiert"]
    vect = TfidfVectorizer()
    X = vect.fit_transform(texts)
    
    st.success(f"✅ TF-IDF OK: {X.shape}")

st.info("💡 Falls das funktioniert, liegt das Problem bei PyTorch/SBERT/spaCy")
