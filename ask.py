"""שאלות על המתכונים משורת הפקודה.
שאלה אחת:  python ask.py "איך מכינים חומוס?"
מצב שיחה:  python ask.py"""
import sys

from src.indexer import load_index
from src.query_engine import ask, format_sources


def answer(index, question: str) -> None:
    response = ask(index, question)
    print(f"\nתשובה: {response}\n\nמקורות:")
    for src in format_sources(response):
        print(f"  [{src['score']}] {src['title']} ({src['category']}, {src['kashrut']})")


def main() -> None:
    index = load_index()
    if len(sys.argv) > 1:
        answer(index, " ".join(sys.argv[1:]))
        return

    while question := input("\nשאלה (Enter ריק ליציאה): ").strip():
        answer(index, question)


if __name__ == "__main__":
    main()
