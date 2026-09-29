"""Build one sklearn Pipeline per (feature set, model).

Everything that learns from data (scaler, TF-IDF vocabulary/IDF) sits inside the
pipeline, so it is only ever fit on training folds.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..features.char_ngrams import make_char_vectorizer
from ..features.handcrafted import GROUPS, HandcraftedFeatures

FEATURE_SETS = ("none", "length", "handcrafted", "char", "combined")
MODELS = ("dummy", "lr", "svm", "rf")


def _handcrafted(cfg: dict, groups=GROUPS) -> HandcraftedFeatures:
    f = cfg["features"]
    return HandcraftedFeatures(
        groups=tuple(groups), function_words=tuple(f["function_words"]), pos_tags=tuple(f["pos_tags"]),
        ttr_window=f["ttr_window"], mattr_window=f["mattr_window"], long_word_len=f["long_word_len"],
        spacy_model=f["spacy_model"], batch_size=f["spacy_batch_size"],
    )


def build_features(feature_set: str, cfg: dict, groups=GROUPS):
    if feature_set == "none":
        return _handcrafted(cfg, ["length"])          # ignored by the dummy model
    if feature_set == "length":
        return Pipeline([("hand", _handcrafted(cfg, ["length"])), ("scale", StandardScaler())])
    if feature_set == "handcrafted":
        return Pipeline([("hand", _handcrafted(cfg, groups)), ("scale", StandardScaler())])
    if feature_set == "char":
        return make_char_vectorizer(cfg["features"]["char_ngrams"])
    if feature_set == "combined":
        hand = Pipeline([("hand", _handcrafted(cfg, groups)), ("scale", StandardScaler())])
        return FeatureUnion(
            [("handcrafted", hand), ("char", make_char_vectorizer(cfg["features"]["char_ngrams"]))],
            transformer_weights={"handcrafted": cfg["features"]["handcrafted_weight"], "char": 1.0},
        )
    raise ValueError(f"unknown feature set {feature_set}")


def build_model(model: str, cfg: dict, seed: int, class_weight=None):
    m = cfg["models"]
    if model == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if model == "lr":
        return LogisticRegression(max_iter=m["lr"]["max_iter"], solver=m["lr"]["solver"],
                                  class_weight=class_weight, random_state=seed)
    if model == "svm":
        return CalibratedClassifierCV(LinearSVC(class_weight=class_weight, random_state=seed, max_iter=10000),
                                      method="sigmoid", cv=m["svm"]["calibration_cv"])
    if model == "rf":
        return RandomForestClassifier(n_estimators=m["rf"]["n_estimators"], class_weight=class_weight,
                                      random_state=seed, n_jobs=-1)
    raise ValueError(f"unknown model {model}")


def build_pipeline(feature_set: str, model: str, cfg: dict, seed: int = 0, groups=GROUPS) -> Pipeline:
    class_weight = "balanced" if cfg["data"].get("balance") == "class_weight" else None
    return Pipeline([
        ("features", build_features(feature_set, cfg, groups)),
        ("clf", build_model(model, cfg, seed, class_weight)),
    ])


def param_grid(feature_set: str, model: str, cfg: dict, tune_analyzer: bool = True) -> dict:
    grid: dict = {}
    if model == "lr":
        grid["clf__C"] = list(cfg["models"]["lr"]["C"])
    elif model == "svm":
        grid["clf__estimator__C"] = list(cfg["models"]["svm"]["C"])
    analyzers = cfg["models"].get("tune_char_analyzer") or []
    if tune_analyzer and len(analyzers) > 1 and model in ("lr", "svm"):
        if feature_set == "char":
            grid["features__analyzer"] = list(analyzers)
        elif feature_set == "combined":
            grid["features__char__analyzer"] = list(analyzers)
    return grid
