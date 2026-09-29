"""Text normalisation. Punctuation and casing are kept on purpose: they are features."""
from __future__ import annotations

import re
import unicodedata

QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
}
_WORD = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")


def strip_header(text: str, header_pattern: str | None) -> str:
    """Drop metadata lines at the top of a file (and blank lines before the essay)."""
    if not header_pattern:
        return text
    header = re.compile(header_pattern)
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or header.match(lines[i])):
        i += 1
    return "\n".join(lines[i:])


def clean_text(text: str, header_pattern: str | None = None) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = strip_header(text, header_pattern)
    for k, v in QUOTE_MAP.items():
        text = text.replace(k, v)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)       # keep paragraph breaks, trim around them
    return text.strip()


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def truncate_words(text: str, n: int | None) -> str:
    """Keep the first n words, cutting the original string so punctuation stays intact."""
    if not n:
        return text
    matches = list(_WORD.finditer(text))
    if len(matches) <= n:
        return text
    end = matches[n - 1].end()
    # keep punctuation right after the last word (e.g. the full stop)
    tail = re.match(r"[^\sA-Za-z0-9]*", text[end:])
    return text[: end + (tail.end() if tail else 0)]
