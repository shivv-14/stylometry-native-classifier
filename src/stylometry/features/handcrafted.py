"""sklearn transformer that combines the five handcrafted feature groups.

spaCy runs once per text (nlp.pipe, batched, NER/lemmatizer disabled). Results are
cached in memory by text hash, and can be persisted to disk, so repeated experiments
do not re-parse. The transformer learns nothing from data, so caching cannot leak
information between train and test.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import function_words as fw_mod
from . import pos as pos_mod
from .function_words import function_word_features
from .pos import pos_features
from .punctuation import NAMES as PUNCT_NAMES
from .punctuation import punctuation_features
from .sentence import NAMES as SENT_NAMES
from .sentence import sentence_features
from .text_utils import get_nlp, word_tokens
from .vocabulary import NAMES as VOCAB_NAMES
from .vocabulary import vocabulary_features

GROUPS = ("sentence", "vocabulary", "function_words", "pos", "punctuation")
GROUP_PREFIX = {"sent_": "sentence", "vocab_": "vocabulary", "fw_": "function_words",
                "pos_": "pos", "punct_": "punctuation", "length_": "length"}

_CACHE: dict[str, dict] = {}


def feature_group(name: str) -> str:
    for prefix, group in GROUP_PREFIX.items():
        if name.startswith(prefix):
            return group
    return "char_ngrams"


def _key(text: str, settings: tuple) -> str:
    return hashlib.sha1((repr(settings) + "\x00" + text).encode("utf-8")).hexdigest()


def load_disk_cache(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    df = pd.read_parquet(path)
    for key, row in df.iterrows():
        _CACHE[key] = row.to_dict()
    return len(df)


def save_disk_cache(path: str | Path) -> None:
    if not _CACHE:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(_CACHE, orient="index").to_parquet(path)


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Text -> DataFrame of stylometric features (normalised per 100 words or proportions).

    groups may contain any of GROUPS, plus "length" (word count + mean sentence length)
    for the length-only baseline.
    """

    def __init__(self, groups=GROUPS, function_words=None, pos_tags=None, ttr_window=100,
                 mattr_window=50, long_word_len=7, spacy_model="en_core_web_sm", batch_size=64):
        self.groups = groups
        self.function_words = function_words
        self.pos_tags = pos_tags
        self.ttr_window = ttr_window
        self.mattr_window = mattr_window
        self.long_word_len = long_word_len
        self.spacy_model = spacy_model
        self.batch_size = batch_size

    # --- helpers -------------------------------------------------------
    def _fw(self):
        return list(self.function_words or fw_mod.DEFAULT_FUNCTION_WORDS)

    def _pos(self):
        return list(self.pos_tags or pos_mod.DEFAULT_POS_TAGS)

    def _settings(self) -> tuple:
        return (tuple(self._fw()), tuple(self._pos()), self.ttr_window, self.mattr_window,
                self.long_word_len, self.spacy_model)

    def group_columns(self, group: str) -> list[str]:
        return {
            "sentence": SENT_NAMES,
            "vocabulary": VOCAB_NAMES,
            "function_words": fw_mod.names(self._fw()),
            "pos": pos_mod.names(self._pos()),
            "punctuation": PUNCT_NAMES,
            "length": ["length_word_count", "sent_mean_len"],
        }[group]

    def columns(self) -> list[str]:
        cols: list[str] = []
        for g in self.groups:
            cols += [c for c in self.group_columns(g) if c not in cols]
        return cols

    def _compute(self, doc, text: str) -> dict:
        n_words = len(word_tokens(doc))
        feats = {"length_word_count": float(n_words)}
        feats.update(sentence_features(doc))
        feats.update(vocabulary_features(doc, self.ttr_window, self.mattr_window, self.long_word_len))
        feats.update(function_word_features(doc, self._fw()))
        feats.update(pos_features(doc, self._pos()))
        feats.update(punctuation_features(text, n_words))
        return feats

    def compute_all(self, texts) -> pd.DataFrame:
        """All features for all groups (cached)."""
        texts = [str(t) for t in texts]
        settings = self._settings()
        keys = [_key(t, settings) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if k not in _CACHE}
        if todo:
            nlp = get_nlp(self.spacy_model)
            items = list(todo.items())
            docs = nlp.pipe((t for _, t in items), batch_size=self.batch_size)
            for (k, t), doc in zip(items, docs):
                _CACHE[k] = self._compute(doc, t)
        return pd.DataFrame([_CACHE[k] for k in keys])

    # --- sklearn API ---------------------------------------------------
    def fit(self, X, y=None):
        self.feature_names_ = np.array(self.columns())
        return self

    def transform(self, X) -> pd.DataFrame:
        df = self.compute_all(X)[self.columns()]
        return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns())
