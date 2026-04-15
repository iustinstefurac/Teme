import json
from pathlib import Path

import chromadb
from openai import OpenAI

from config import (
    BOOK_DATA_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    OPENAI_EMBEDDING_MODEL,
    validate_settings,
)


def load_books() -> list[dict]:
    with open(BOOK_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_retrieval_text(book: dict) -> str:
    themes = ", ".join(book["themes"])
    return (
        f"Title: {book['title']}\n"
        f"Author: {book['author']}\n"
        f"Themes: {themes}\n"
        f"Short summary: {book['short_summary']}"
    )


def create_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def main() -> None:
    validate_settings()

    books = load_books()
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete old collection if it exists, so repeated runs do not duplicate data.
    try:
        chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"description": "Smart Librarian book summaries"}
    )

    ids = []
    documents = []
    metadatas = []
    retrieval_texts = []

    for book in books:
        retrieval_text = build_retrieval_text(book)

        ids.append(book["id"])
        documents.append(retrieval_text)
        retrieval_texts.append(retrieval_text)
        metadatas.append(
            {
                "title": book["title"],
                "author": book["author"],
                "themes": ", ".join(book["themes"]),
                "short_summary": book["short_summary"],
            }
        )

    embeddings = create_embeddings(openai_client, retrieval_texts)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {collection.count()} books into collection '{CHROMA_COLLECTION_NAME}'.")
    print(f"Chroma database saved in: {CHROMA_PATH}")


if __name__ == "__main__":
    main()