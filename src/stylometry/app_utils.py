"""Helpers for the Streamlit app, kept here so they can be unit-tested."""
from __future__ import annotations

from dataclasses import dataclass

from .data.clean import word_count


@dataclass
class Validation:
    ok: bool
    level: str        # "error", "warning" or "ok"
    message: str
    words: int


def validate_text(text: str, min_words: int = 40, warn_words: int = 120) -> Validation:
    n = word_count(text or "")
    if n == 0:
        return Validation(False, "error", "Please paste some English text.", n)
    if n < min_words:
        return Validation(False, "error",
                          f"The text has {n} words. At least {min_words} are needed for any estimate.", n)
    if n < warn_words:
        return Validation(True, "warning",
                          f"The text has {n} words. Below {warn_words} words the style statistics are "
                          "noisy and the model confidence is less reliable.", n)
    return Validation(True, "ok", f"{n} words.", n)


STYLE_PANEL = {
    "sent_mean_len": "Average sentence length (words)",
    "vocab_mattr": "Vocabulary diversity (MATTR, window 50)",
    "vocab_avg_word_len": "Average word length (characters)",
    "pos_NOUN": "Nouns (% of words)",
    "pos_VERB": "Verbs (% of words)",
    "punct_comma": "Commas per 100 words",
    "fw_total": "Function words per 100 words",
}


def style_summary(features_row) -> list[tuple[str, float]]:
    return [(label, float(features_row[key])) for key, label in STYLE_PANEL.items() if key in features_row]
