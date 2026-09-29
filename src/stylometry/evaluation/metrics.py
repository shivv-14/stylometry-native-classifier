"""Classification metrics and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

LABELS = ["native", "non_native"]


def compute_metrics(y_true, y_pred, y_score=None, labels=LABELS, positive: str = "native") -> dict:
    """y_score: probability of the positive class (or None)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    for i, lab in enumerate(labels):
        out[f"precision_{lab}"] = float(p[i])
        out[f"recall_{lab}"] = float(r[i])
        out[f"f1_{lab}"] = float(f[i])
    if y_score is not None and len(set(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true == positive, y_score)
    else:
        out["roc_auc"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()
    return out


def bootstrap_f1_ci(y_true, y_pred, n: int = 1000, seed: int = 0, alpha: float = 0.05,
                    labels=LABELS) -> tuple[float, float]:
    """95% percentile bootstrap CI for macro-F1 over test documents."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    scores = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        scores.append(f1_score(y_true[s], y_pred[s], labels=labels, average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def positive_scores(model, X, positive: str = "native"):
    """Probability of the positive class, or None if the model has no predict_proba."""
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    if positive not in classes:
        return None
    return model.predict_proba(X)[:, classes.index(positive)]
