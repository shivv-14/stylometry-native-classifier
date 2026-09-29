"""Run one experiment configuration over several seeds and aggregate the results."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.split import assert_no_writer_overlap, document_split, topic_split, writer_split
from ..models.train import fit_tuned
from .metrics import bootstrap_f1_ci, compute_metrics, positive_scores

SUMMARY_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc",
                   "f1_native", "f1_non_native", "recall_native", "recall_non_native",
                   "precision_native", "precision_non_native", "f1_ci_low", "f1_ci_high", "cv_f1_macro"]


@dataclass
class RunResult:
    row: dict
    pipeline: object
    test: pd.DataFrame
    y_pred: np.ndarray
    y_score: np.ndarray | None
    extra: dict = field(default_factory=dict)


def make_split(df: pd.DataFrame, kind: str, cfg: dict, seed: int, train_topic: str | None = None):
    test_size = cfg["split"]["test_size"]
    if kind == "writer":
        train, test = writer_split(df, test_size, seed)
        assert_no_writer_overlap(train, test)
    elif kind == "document":
        train, test = document_split(df, test_size, seed)
    elif kind == "topic":
        train, test = topic_split(df, train_topic, seed)
        assert_no_writer_overlap(train, test)
    else:
        raise ValueError(kind)
    return train, test


def run_once(exp_id: str, feature_set: str, model: str, df: pd.DataFrame, cfg: dict, seed: int,
             split: str = "writer", feature_groups=None, variant: str = "", train_topic: str | None = None,
             tune_analyzer: bool = True) -> RunResult:
    train, test = make_split(df, split, cfg, seed, train_topic)
    pos = cfg["data"]["positive_label"]
    pipe, params, cv_f1 = fit_tuned(feature_set, model, train, cfg, seed, feature_groups, tune_analyzer)
    y_pred = pipe.predict(test["text"])
    y_score = positive_scores(pipe, test["text"], pos)
    met = compute_metrics(test["label"], y_pred, y_score, positive=pos)
    lo, hi = bootstrap_f1_ci(test["label"], y_pred, cfg["evaluation"]["bootstrap_samples"], seed)
    row = {
        "exp_id": exp_id, "feature_set": feature_set, "model": model, "split": split,
        "variant": variant, "seed": seed, "n_train": len(train), "n_test": len(test),
        "train_writers": train.writer_id.nunique(), "test_writers": test.writer_id.nunique(),
        "cv_f1_macro": cv_f1 if cv_f1 is not None else np.nan,
        "f1_ci_low": lo, "f1_ci_high": hi, "params": str(params),
        **{k: v for k, v in met.items() if k != "confusion_matrix"},
        "confusion_matrix": str(met["confusion_matrix"]),
    }
    return RunResult(row, pipe, test, y_pred, y_score)


def run_seeds(exp_id: str, feature_set: str, model: str, df: pd.DataFrame, cfg: dict, seeds,
              **kwargs) -> list[RunResult]:
    results = []
    for s in seeds:
        r = run_once(exp_id, feature_set, model, df, cfg, s, **kwargs)
        print(f"  {exp_id:<3} {feature_set:<12} {model:<5} {r.row['split']:<8} {r.row['variant']:<22} "
              f"seed={s} macro-F1={r.row['f1_macro']:.3f}", flush=True)
        results.append(r)
    return results


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """mean and std over seeds for every (experiment, feature set, model, split, variant)."""
    keys = ["exp_id", "feature_set", "model", "split", "variant"]
    metrics = [m for m in SUMMARY_METRICS if m in runs.columns]
    g = runs.groupby(keys, sort=False, dropna=False)[metrics]
    mean, std = g.mean().add_suffix("_mean"), g.std(ddof=0).add_suffix("_std")
    out = pd.concat([mean, std], axis=1)
    out["n_seeds"] = g.size()
    out["n_test_mean"] = runs.groupby(keys, sort=False, dropna=False)["n_test"].mean()
    ordered = ["n_seeds", "n_test_mean"] + [c for m in metrics for c in (f"{m}_mean", f"{m}_std")]
    return out[ordered].reset_index()
