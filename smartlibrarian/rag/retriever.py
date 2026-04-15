import chromadb
from openai import OpenAI

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    OPENAI_EMBEDDING_MODEL,
    validate_settings,
)


def create_query_embedding(user_query: str) -> list[float]:
    validate_settings()
    client = OpenAI()

    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=user_query,
    )
    return response.data[0].embedding


def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Chroma collection not found. Run `python -m rag.index_books` first."
        ) from exc


def search_books(user_query: str, top_k: int = 3) -> list[dict]:
    collection = get_collection()
    query_embedding = create_query_embedding(user_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []

    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        matches.append(
            {
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )

    return matches