"""ממשק Streamlit לחיפוש מתכונים. הרצה: streamlit run app.py"""
import json

import streamlit as st
from google.genai.errors import APIError

from src.config import DATA_FILE, TOP_K
from src.indexer import load_index
from src.query_engine import ask, format_sources

ALL = "הכול"

st.set_page_config(page_title="חיפוש מתכונים", page_icon="🍲")
# Streamlit מוצג משמאל לימין; מיישרים לימין בשביל עברית
st.markdown("<style>.stApp { direction: rtl; text-align: right; }</style>", unsafe_allow_html=True)
st.title("🍲 חיפוש מתכונים חכם")
st.caption("מערכת RAG על מאגר מתכונים: LlamaIndex + Chroma + Gemini")


@st.cache_resource
def get_index():
    """נטען פעם אחת בלבד, ולא בכל לחיצה על כפתור."""
    return load_index()


@st.cache_data
def get_options() -> tuple[list[str], list[str]]:
    """רשימות הקטגוריות והכשרויות נלקחות מהמאגר עצמו."""
    with open(DATA_FILE, encoding="utf-8") as f:
        recipes = json.load(f)
    categories = sorted({r["category"] for r in recipes})
    kashrut = sorted({r["kashrut"] for r in recipes})
    return categories, kashrut


categories, kashrut_types = get_options()

with st.sidebar:
    st.header("הגדרות חיפוש")
    top_k = st.slider("מספר מקטעים לשליפה (top_k)", 1, 8, TOP_K)
    category = st.selectbox("קטגוריה", [ALL] + categories)
    kashrut = st.selectbox("כשרות", [ALL] + kashrut_types)

try:
    index = get_index()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

question = st.text_input("מה בא לך לבשל?", placeholder="למשל: מה אפשר להכין עם עדשים?")

if question:
    try:
        with st.spinner("מחפש במאגר..."):
            response = ask(
                index, question, top_k=top_k,
                category=None if category == ALL else category,
                kashrut=None if kashrut == ALL else kashrut,
            )
    except APIError as e:
        # 429 = נגמרה המכסה היומית, 503 = המודל עמוס
        if e.code == 429:
            st.error("נגמרה המכסה היומית של המודל. אפשר להחליף מודל ב-LLM_MODEL בקובץ .env או לנסות שוב מחר.")
        elif e.code == 503:
            st.error("המודל עמוס כרגע. נסי שוב בעוד דקה או החליפי מודל בקובץ .env.")
        else:
            st.error(f"שגיאה מ-Gemini ({e.code}): {e.message}")
        st.stop()

    st.subheader("תשובה")
    st.write(str(response))

    st.subheader("מקורות")
    sources = format_sources(response)
    if not sources:
        st.info("לא נמצאו מקטעים שעומדים בסינון.")
    for src in sources:
        with st.expander(f"{src['title']} · {src['category']} · {src['kashrut']} · ציון {src['score']}"):
            st.text(src["text"])
