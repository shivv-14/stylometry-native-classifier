"""Tune and train the final pipelines and save them to models/.

    python -m stylometry.models.train

Model selection uses cross-validation inside the training writers only; the
held-out writers are touched once, to report test metrics in metadata.json.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from ..config import load_config, rel, resolve, set_seed
from ..data.split import assert_no_writer_overlap, cv_splitter, writer_split
from ..evaluation.metrics import compute_metrics, positive_scores
from ..features.handcrafted import load_disk_cache, save_disk_cache
from .pipelines import build_pipeline, param_grid

FINAL_FEATURE_SETS = ("handcrafted", "char", "combined")
FINAL_MODELS = ("lr", "svm")


def n_cv_folds(y: pd.Series, groups: pd.Series, wanted: int) -> int:
    """Can't have more folds than writers in the smallest class."""
    writers_per_class = pd.DataFrame({"y": y, "g": groups}).drop_duplicates().groupby("y").size()
    return max(2, min(wanted, int(writers_per_class.min())))


def fit_tuned(feature_set: str, model: str, train: pd.DataFrame, cfg: dict, seed: int,
              feature_groups=None, tune_analyzer: bool = True):
    """Return (fitted pipeline, best params, best CV macro-F1 or None)."""
    kwargs = {} if feature_groups is None else {"groups": tuple(feature_groups)}
    pipe = build_pipeline(feature_set, model, cfg, seed, **kwargs)
    grid = param_grid(feature_set, model, cfg, tune_analyzer)
    X, y, g = train["text"], train["label"], train["writer_id"]
    if not grid:
        pipe.fit(X, y)
        return pipe, {}, None
    folds = n_cv_folds(y, g, cfg["split"]["cv_folds"])
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv_splitter(folds, seed), n_jobs=1, refit=True)
    gs.fit(X, y, groups=g)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_processed(cfg: dict) -> pd.DataFrame:
    path = resolve(cfg, "processed")
    if not path.exists():
        print(f"{rel(path)} not found. Run `python tasks.py prepare` first "
              "(it explains how to obtain the corpus).")
        sys.exit(0)
    return pd.read_parquet(path)


def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    seed = cfg["seed"]
    set_seed(seed)
    df = load_processed(cfg)
    cache = resolve(cfg, "feature_cache")
    load_disk_cache(cache)

    train, test = writer_split(df, cfg["split"]["test_size"], seed)
    assert_no_writer_overlap(train, test)
    out_dir = resolve(cfg, "models_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = cfg["data"]["positive_label"]

    runs = {}
    for fs in FINAL_FEATURE_SETS:
        for m in FINAL_MODELS:
            print(f"training {fs}/{m} ...", flush=True)
            pipe, params, cv_f1 = fit_tuned(fs, m, train, cfg, seed)
            pred = pipe.predict(test["text"])
            met = compute_metrics(test["label"], pred, positive_scores(pipe, test["text"], pos), positive=pos)
            name = f"{fs}_{m}"
            joblib.dump(pipe, out_dir / f"{name}.joblib")
            runs[name] = {"feature_set": fs, "model": m, "params": {k: str(v) for k, v in params.items()},
                          "cv_f1_macro": cv_f1, "test_metrics": met}
            print(f"  cv macro-F1 {cv_f1:.3f} | test macro-F1 {met['f1_macro']:.3f}")
            save_disk_cache(cache)

    # best feature set = highest mean CV macro-F1 over LR and SVM (training data only)
    cv_by_fs = {fs: sum(runs[f"{fs}_{m}"]["cv_f1_macro"] for m in FINAL_MODELS) / len(FINAL_MODELS)
                for fs in FINAL_FEATURE_SETS}
    best = max(cv_by_fs, key=cv_by_fs.get)

    counts = df.groupby("label").agg(texts=("text", "size"), writers=("writer_id", "nunique"))
    meta = {
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "path": cfg["paths"]["processed"], "sha256": file_hash(resolve(cfg, "processed")),
            "source": cfg["data"]["source"], "counts": counts.to_dict(orient="index"),
            "train_writers": int(train.writer_id.nunique()), "test_writers": int(test.writer_id.nunique()),
            "train_texts": len(train), "test_texts": len(test),
        },
        "split": {"type": "writer-separated", "seed": seed, "test_size": cfg["split"]["test_size"]},
        "best_feature_set": best,
        "cv_f1_by_feature_set": cv_by_fs,
        "models": runs,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"best feature set (CV on training writers): {best}")
    print(f"saved models and metadata to {rel(out_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
