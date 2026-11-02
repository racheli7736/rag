#"""טעינת המתכונים מקובץ JSON והפיכתם ל-Documents של LlamaIndex."""
import json

from llama_index.core import Document

from src.config import DATA_FILE


def recipe_to_text(recipe: dict) -> str:
   # """בונה טקסט קריא מהמתכון – זה הטקסט שעובר Chunking ו-Embedding."""
    ingredients = "\n".join(f"- {item}" for item in recipe["ingredients"])
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(recipe["instructions"], 1))
    return (
        f"מתכון: {recipe['title']}\n"
        f"קטגוריה: {recipe['category']} | כשרות: {recipe['kashrut']} | "
        f"זמן הכנה: {recipe['prep_time_minutes']} דקות | רמת קושי: {recipe['difficulty']}\n"
        f"תגיות: {', '.join(recipe['tags'])}\n\n"
        f"מצרכים:\n{ingredients}\n\n"
        f"אופן ההכנה:\n{steps}\n\n"
        f"טיפים: {recipe.get('tips', '')}"
    )


def load_recipes() -> list[Document]:
    with open(DATA_FILE, encoding="utf-8") as f:
        recipes = json.load(f)

    documents = []
    for recipe in recipes:
        # Chroma מקבל רק ערכי metadata פשוטים (str/int/float/bool), לכן התגיות נשמרות כמחרוזת
        metadata = {
            "recipe_id": recipe["id"],
            "title": recipe["title"],
            "category": recipe["category"],
            "kashrut": recipe["kashrut"],
            "prep_time_minutes": recipe["prep_time_minutes"],
            "difficulty": recipe["difficulty"],
            "tags": ", ".join(recipe["tags"]),
        }
        documents.append(Document(text=recipe_to_text(recipe), metadata=metadata, id_=recipe["id"]))
    return documents


if __name__ == "__main__":
    docs = load_recipes()
    print(f"נטענו {len(docs)} מתכונים\n")
    print(docs[0].text)
    print("\nmetadata:", docs[0].metadata)
