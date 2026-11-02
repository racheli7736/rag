"""הגדרות מרכזיות של הפרויקט – נטענות מקובץ .env."""
import os
from pathlib import Path

import truststore
from dotenv import load_dotenv

# NetFree מחליף את תעודות ה-SSL; משתמשים במאגר התעודות של Windows כדי ש-Python יסמוך עליהן
truststore.inject_into_ssl()

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_FILE = ROOT_DIR / "data" / "recipes.json"
CHROMA_DIR = ROOT_DIR / "storage" / "chroma"
COLLECTION_NAME = "recipes"

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))


def configure_models() -> None:
    """מגדיר ל-LlamaIndex באיזה LLM ובאיזה מודל Embeddings להשתמש."""
    from google.genai import types
    from llama_index.core import Settings
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("חסר GOOGLE_API_KEY – מלאי אותו בקובץ .env")

    # thinking_level="low" – בלי "חשיבה" ארוכה; לשאלות על מתכונים זה מספיק והרבה יותר מהיר
    Settings.llm = GoogleGenAI(
        model=LLM_MODEL,
        generation_config=types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL)
