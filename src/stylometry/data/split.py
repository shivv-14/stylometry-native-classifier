"""Train/test splits. The writer-separated split is the main protocol."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


class WriterLeakageError(AssertionError):
    pass


def assert_no_writer_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared = set(train["writer_id"]) & set(test["writer_id"])
    if shared:
        raise WriterLeakageError(f"{len(shared)} writers appear in both train and test, e.g. {sorted(shared)[:3]}")


def writer_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    """About `test_size` of the writers of each class go to test; no writer is in both sets."""
    n_splits = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["writer_id"]))
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    assert_no_writer_overlap(train, test)
    return train, test


def document_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    """Random split over documents. Writers can leak across sets; used only as a contrast."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def topic_split(df: pd.DataFrame, train_topic: str, seed: int = 0):
    """Train on one prompt, test on the other prompt(s).

    In ICNALE every writer answers both prompts, so writers are first split in half:
    train = train_topic essays of half A, test = other-topic essays of half B.
    """
    a, b = writer_split(df, test_size=0.5, seed=seed)
    train = a[a.topic == train_topic]
    test = b[b.topic != train_topic]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def cv_splitter(n_splits: int, seed: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def class_ratio(df: pd.DataFrame, positive: str = "native") -> float:
    return float(np.mean(df["label"] == positive))
