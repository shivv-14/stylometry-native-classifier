"""Figures written to reports/figures/.

matplotlib (PNG) is used when it can be loaded; otherwise the SVG versions in
svg_plots.py are written instead (same content, no compiled dependencies).
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from . import svg_plots
from .metrics import LABELS

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
    HAVE_MATPLOTLIB = True
except (ImportError, OSError) as e:          # e.g. DLL blocked by an application-control policy
    HAVE_MATPLOTLIB = False
    print(f"matplotlib unavailable ({e.__class__.__name__}); writing SVG figures instead")


def _fallback(fn):
    """Use the SVG implementation of the same name when matplotlib is missing."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        impl = fn if HAVE_MATPLOTLIB else getattr(svg_plots, fn.__name__)
        return impl(*args, **kwargs)
    return wrapper


CLASS_COLORS = {"native": "#2a6fdb", "non_native": "#e07b39"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@_fallback
def confusion_matrix_plot(y_true, y_pred, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    _save(fig, path)


@_fallback
def roc_plot(curves: list[tuple[str, np.ndarray, np.ndarray]], path: Path, positive: str = "native"):
    """curves: (name, y_true, positive-class score)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, y_true, score in curves:
        RocCurveDisplay.from_predictions(np.asarray(y_true) == positive, score, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_title(f"ROC (positive class: {positive}, seed 0 split)", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path)


@_fallback
def top_features_plot(top: pd.DataFrame, title: str, path: Path, k: int = 20):
    if top.empty:
        return
    labels = list(dict.fromkeys(top["direction"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.28 * k + 1.5))
    for ax, lab in zip(axes, labels):
        d = top[top.direction == lab].copy()
        d = d.reindex(d.coef.abs().sort_values().index).tail(k)
        ax.barh(d.feature.astype(str), d.coef, color=CLASS_COLORS.get(lab, "grey"))
        ax.set_title(f"toward {lab}", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.axvline(0, c="k", lw=0.6)
    fig.suptitle(title, fontsize=10)
    _save(fig, path)


@_fallback
def ablation_plot(abl: pd.DataFrame, path: Path):
    """abl: columns group, f1_drop_mean, f1_drop_std."""
    d = abl.sort_values("f1_drop_mean")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(d.group, d.f1_drop_mean, xerr=d.f1_drop_std, color="#6c5ce7", capsize=3)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("drop in macro-F1 when the group is removed (mean ± std over seeds)")
    ax.set_title("Feature-group ablation (LR, handcrafted, writer-separated)", fontsize=10)
    _save(fig, path)


@_fallback
def distribution_plots(features: pd.DataFrame, labels: pd.Series, columns: list[str], path: Path):
    cols = [c for c in columns if c in features.columns]
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        for lab in LABELS:
            v = features.loc[labels.values == lab, c]
            ax.hist(v, bins=30, alpha=0.55, density=True, label=lab, color=CLASS_COLORS[lab])
        ax.set_title(c, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Feature distributions by class (full processed dataset)", fontsize=10)
    _save(fig, path)
