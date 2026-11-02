"""Retrieval + Generation: שליפת מקטעים רלוונטיים ויצירת תשובה עם מקורות."""
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

from src.config import TOP_K

NOT_FOUND = "לא מצאתי מתכון מתאים במאגר."

QA_PROMPT = PromptTemplate(
    "את/ה עוזר/ת מטבח שעונה רק על סמך המתכונים שמופיעים בהקשר.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "כללים:\n"
    "1. ענה/י בעברית, בקצרה ובבהירות.\n"
    f"2. השתמש/י רק במידע מההקשר. אם התשובה לא נמצאת בו, כתוב/כתבי בדיוק: \"{NOT_FOUND}\"\n"
    "3. ציין/צייני את שם המתכון שעליו מבוססת התשובה.\n"
    "4. ההקשר הוא רק מדגם קטן של מתכונים שנשלפו לפי דמיון לשאלה, ולא כל המאגר. "
    "לכן לעולם אל תתאר/י אותם כ\"המתכונים במאגר\" ואל תציין/צייני כמה מתכונים יש. "
    "כשמציעים כמה מתכונים, פתח/י במשפט: \"הנה כמה מתכונים שמתאימים לבקשה:\"\n\n"
    "שאלה: {query_str}\n"
    "תשובה: "
)


def build_filters(category: str | None = None, kashrut: str | None = None) -> MetadataFilters | None:
    """Metadata Filtering – Chroma מחפש רק בין מקטעים שעומדים בתנאים."""
    filters = []
    if category:
        filters.append(MetadataFilter(key="category", value=category))
    if kashrut:
        filters.append(MetadataFilter(key="kashrut", value=kashrut))
    return MetadataFilters(filters=filters) if filters else None


def ask(index: VectorStoreIndex, question: str, top_k: int = TOP_K,
        category: str | None = None, kashrut: str | None = None):
    """שולף top_k מקטעים (אחרי סינון), מכניס אותם ל-prompt ושולח ל-Gemini."""
    engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=QA_PROMPT,
        filters=build_filters(category, kashrut),
    )
    return engine.query(question)


def format_sources(response) -> list[dict]:
    """מחלץ מהתשובה את המקטעים שעליהם התבססה – להצגת מקורות."""
    return [
        {
            "title": s.node.metadata.get("title"),
            "category": s.node.metadata.get("category"),
            "kashrut": s.node.metadata.get("kashrut"),
            "score": round(s.score or 0.0, 3),
            "text": s.node.get_content(),
        }
        for s in response.source_nodes
    ]
