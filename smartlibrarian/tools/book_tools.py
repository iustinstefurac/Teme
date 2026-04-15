import json
from functools import lru_cache

from config import BOOK_DATA_PATH #


def normalize_title(title: str) -> str: # Normalize by stripping, lowercasing, and collapsing whitespace
    return " ".join(title.strip().lower().split())


@lru_cache(maxsize=1) # Cache the loaded books in memory for faster access on subsequent calls
def load_books() -> list[dict]:
    with open(BOOK_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_summary_by_title(title: str) -> str:
    normalized_input = normalize_title(title)

    for book in load_books():
        if normalize_title(book["title"]) == normalized_input: # Use normalized comparison for more forgiving matching
            return book["full_summary"]

    available_titles = ", ".join(book["title"] for book in load_books())
    return (
        f"Title not found for exact match: {title}. "
        f"Available exact titles are: {available_titles}"
    )