"""שלב ההכנה (Offline): בניית האינדקס. מריצים פעם אחת ובכל שינוי במתכונים."""
from src.indexer import build_index

if __name__ == "__main__":
    build_index()
    print("האינדקס נבנה ונשמר בתיקייה storage/chroma")
