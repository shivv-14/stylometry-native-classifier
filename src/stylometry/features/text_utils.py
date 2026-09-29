"""Shared helpers: spaCy loading and token selection."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def get_nlp(model: str = "en_core_web_sm"):
    import spacy

    try:
        return spacy.load(model, disable=["ner", "lemmatizer"])
    except OSError as e:
        raise OSError(f"spaCy model '{model}' is missing. Run: python -m spacy download {model}") from e


def word_tokens(span) -> list:
    """Tokens that count as words: not punctuation, not whitespace."""
    return [t for t in span if not t.is_punct and not t.is_space]


def alpha_words(span) -> list[str]:
    return [t.text.lower() for t in span if t.is_alpha]
