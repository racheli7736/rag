"""בניית האינדקס הווקטורי (Chroma) וטעינתו מחדש."""
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, configure_models
from src.data_loader import load_recipes


def get_collection(reset: bool = False):
    """מחזיר את ה-collection של Chroma מהדיסק. reset=True מוחק ובונה מאפס."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    return client.get_or_create_collection(COLLECTION_NAME)


def build_index() -> VectorStoreIndex:
    """מסמכים ← Nodes ← Embeddings ← שמירה ב-Chroma."""
    configure_models()
    documents = load_recipes()

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Documents: {len(documents)}  Nodes: {len(nodes)}")

    vector_store = ChromaVectorStore(chroma_collection=get_collection(reset=True))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)


def load_index() -> VectorStoreIndex:
    """טוען אינדקס קיים מהדיסק בלי לחשב Embeddings מחדש."""
    configure_models()
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("האינדקס ריק – הריצי קודם: python build_index.py")
    return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection))
