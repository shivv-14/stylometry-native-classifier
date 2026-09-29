"""Explanations for linear models.

Wording rule: these are features *associated with the model's prediction on this
dataset*, not causes of native or non-native writing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..features.handcrafted import feature_group


def linear_coefficients(pipe) -> np.ndarray | None:
    """Coefficient vector for the 'native' direction (positive = pushes toward classes_[1]).

    For CalibratedClassifierCV(LinearSVC) the coefficients of the calibrated folds are averaged.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return np.asarray(clf.coef_).ravel()
    if hasattr(clf, "calibrated_classifiers_"):
        coefs = [np.asarray(c.estimator.coef_).ravel() for c in clf.calibrated_classifiers_
                 if hasattr(c.estimator, "coef_")]
        if coefs:
            return np.mean(coefs, axis=0)
    return None


def feature_names(pipe) -> np.ndarray:
    feats = pipe.named_steps["features"]
    names = np.asarray(feats.get_feature_names_out(), dtype=object)
    # FeatureUnion prefixes names with the transformer name; the char vectorizer names are raw n-grams
    clean = []
    for n in names:
        n = str(n)
        if n.startswith("handcrafted__"):
            n = n[len("handcrafted__"):]
        elif n.startswith("char__"):
            n = "char:" + repr(n[len("char__"):])
        elif feature_group(n) == "char_ngrams":
            n = "char:" + repr(n)
        clean.append(n)
    return np.array(clean, dtype=object)


def class_labels(pipe) -> tuple[str, str]:
    """(label pushed by negative coefficients, label pushed by positive coefficients)."""
    c = list(pipe.named_steps["clf"].classes_)
    return c[0], c[1]


def top_features(pipe, k: int = 20) -> pd.DataFrame:
    """Global top-k coefficients toward each class."""
    coef = linear_coefficients(pipe)
    if coef is None:
        return pd.DataFrame(columns=["feature", "group", "coef", "direction"])
    names = feature_names(pipe)
    neg_label, pos_label = class_labels(pipe)
    df = pd.DataFrame({"feature": names, "coef": coef})
    df["group"] = [group_of(n) for n in names]
    top_pos = df.nlargest(k, "coef").assign(direction=pos_label)
    top_neg = df.nsmallest(k, "coef").assign(direction=neg_label)
    return pd.concat([top_pos, top_neg], ignore_index=True)


def group_of(name: str) -> str:
    return "char_ngrams" if name.startswith("char:") else feature_group(name)


def coef_group_importance(pipe) -> pd.DataFrame:
    """Sum of |coef| per handcrafted group (inputs are standardised, so magnitudes are comparable)."""
    coef = linear_coefficients(pipe)
    if coef is None:
        return pd.DataFrame(columns=["group", "abs_coef_sum"])
    names = feature_names(pipe)
    df = pd.DataFrame({"group": [group_of(n) for n in names], "abs_coef_sum": np.abs(coef)})
    return df.groupby("group", as_index=False)["abs_coef_sum"].sum().sort_values("abs_coef_sum", ascending=False)


def local_contributions(pipe, text: str, k: int = 8) -> pd.DataFrame:
    """Per-feature contribution = coefficient x (standardised / tf-idf) feature value for one text.

    Returns the top k features pushing toward each class.
    """
    coef = linear_coefficients(pipe)
    if coef is None:
        return pd.DataFrame(columns=["feature", "group", "value", "contribution", "toward"])
    x = pipe.named_steps["features"].transform(pd.Series([text]))
    x = x.toarray() if sparse.issparse(x) else np.asarray(x, dtype=float)
    x = x.ravel()
    contrib = coef * x
    names = feature_names(pipe)
    neg_label, pos_label = class_labels(pipe)
    df = pd.DataFrame({"feature": names, "group": [group_of(n) for n in names], "value": x,
                       "contribution": contrib})
    df = df[df.contribution != 0]
    pos = df.nlargest(k, "contribution").assign(toward=pos_label)
    neg = df.nsmallest(k, "contribution").assign(toward=neg_label)
    neg = neg[neg.contribution < 0]
    pos = pos[pos.contribution > 0]
    return pd.concat([pos, neg], ignore_index=True)
