"""Dependency-free SVG versions of the report figures.

Used by plots.py when matplotlib cannot be loaded (e.g. its compiled extension is
blocked by an OS application-control policy). Same inputs, .svg output.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

LABELS = ["native", "non_native"]
CLASS_COLORS = {"native": "#2a6fdb", "non_native": "#e07b39"}
SERIES = ["#2a6fdb", "#e07b39", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
FONT = "font-family='Helvetica, Arial, sans-serif'"


class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill, opacity=1.0, stroke="none"):
        self.items.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 0):.1f}' height='{max(h, 0):.1f}' "
                          f"fill='{fill}' fill-opacity='{opacity}' stroke='{stroke}'/>")

    def line(self, x1, y1, x2, y2, stroke="#333", width=1.0, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.items.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                          f"stroke='{stroke}' stroke-width='{width}'{d}/>")

    def polyline(self, pts, stroke, width=2.0):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.items.append(f"<polyline points='{p}' fill='none' stroke='{stroke}' stroke-width='{width}'/>")

    def text(self, x, y, s, size=11, anchor="start", weight="normal", fill="#222", rotate=None):
        rot = f" transform='rotate({rotate} {x:.1f} {y:.1f})'" if rotate else ""
        self.items.append(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' text-anchor='{anchor}' "
                          f"font-weight='{weight}' fill='{fill}' {FONT}{rot}>{escape(str(s))}</text>")

    def save(self, path: Path) -> Path:
        path = Path(path).with_suffix(".svg")
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.items)
        path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' "
                        f"viewBox='0 0 {self.w} {self.h}'>\n<rect width='100%' height='100%' fill='white'/>\n"
                        f"{body}\n</svg>\n", encoding="utf-8", newline="\n")
        return path


def _ticks(lo, hi, n=5):
    return np.linspace(lo, hi, n)


def confusion_matrix_plot(y_true, y_pred, title: str, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    s = Svg(360, 330)
    s.text(180, 22, title, 13, "middle", "bold")
    x0, y0, c = 110, 50, 100
    vmax = max(cm.max(), 1)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            s.rect(x0 + j * c, y0 + i * c, c, c, "#2a6fdb", 0.1 + 0.8 * v / vmax, "#fff")
            s.text(x0 + j * c + c / 2, y0 + i * c + c / 2 + 6, v, 18, "middle", "bold",
                   "#fff" if v / vmax > 0.5 else "#222")
        s.text(x0 - 8, y0 + i * c + c / 2 + 4, LABELS[i], 11, "end")
        s.text(x0 + i * c + c / 2, y0 + 2 * c + 18, LABELS[i], 11, "middle")
    s.text(x0 + c, y0 + 2 * c + 40, "Predicted label", 11, "middle")
    s.text(24, y0 + c, "True label", 11, "middle", rotate=-90)
    s.save(path)


def roc_plot(curves, path: Path, positive: str = "native"):
    s = Svg(560, 470)
    s.text(280, 22, f"ROC (positive class: {positive}, seed 0 split)", 13, "middle", "bold")
    x0, y0, w, h = 60, 40, 340, 340
    s.rect(x0, y0, w, h, "none", stroke="#333")
    for t in _ticks(0, 1, 6):
        s.line(x0 + t * w, y0 + h, x0 + t * w, y0 + h + 4)
        s.text(x0 + t * w, y0 + h + 17, f"{t:.1f}", 10, "middle")
        s.line(x0 - 4, y0 + h - t * h, x0, y0 + h - t * h)
        s.text(x0 - 7, y0 + h - t * h + 4, f"{t:.1f}", 10, "end")
    s.line(x0, y0 + h, x0 + w, y0, "#999", 1, "4,4")
    s.text(x0 + w / 2, y0 + h + 36, "False positive rate", 11, "middle")
    s.text(18, y0 + h / 2, "True positive rate", 11, "middle", rotate=-90)
    for k, (name, y_true, score) in enumerate(curves):
        fpr, tpr, _ = roc_curve(np.asarray(y_true) == positive, score)
        color = SERIES[k % len(SERIES)]
        s.polyline([(x0 + a * w, y0 + h - b * h) for a, b in zip(fpr, tpr)], color)
        ly = y0 + 10 + k * 18
        s.line(x0 + w + 12, ly - 4, x0 + w + 32, ly - 4, color, 3)
        s.text(x0 + w + 36, ly, f"{name} (AUC {auc(fpr, tpr):.2f})", 10)
    s.save(path)


def _barh_panel(s: Svg, names, values, colors, x0, y0, w, row, errors=None, title=None):
    values = np.asarray(values, dtype=float)
    lo, hi = min(0.0, float(values.min())), max(0.0, float(values.max()))
    if errors is not None:
        lo = min(lo, float((values - errors).min()))
        hi = max(hi, float((values + errors).max()))
    span = (hi - lo) or 1.0
    to_x = lambda v: x0 + (v - lo) / span * w  # noqa: E731
    if title:
        s.text(x0 + w / 2, y0 - 8, title, 12, "middle", "bold")
    for i, (n, v, c) in enumerate(zip(names, values, colors)):
        y = y0 + i * row
        a, b = sorted((to_x(0), to_x(v)))
        s.rect(a, y + 2, b - a, row - 4, c)
        s.text(x0 - 6, y + row / 2 + 4, n, 10, "end")
        if errors is not None:
            e = errors[i]
            s.line(to_x(v - e), y + row / 2, to_x(v + e), y + row / 2, "#222", 1.2)
    s.line(to_x(0), y0, to_x(0), y0 + len(names) * row, "#222", 0.8)
    for t in _ticks(lo, hi, 5):
        s.text(to_x(t), y0 + len(names) * row + 14, f"{t:.2f}", 9, "middle")


def top_features_plot(top: pd.DataFrame, title: str, path: Path, k: int = 20):
    if top.empty:
        return
    labels = list(dict.fromkeys(top["direction"]))
    row = 18
    s = Svg(1100, 90 + k * row)
    s.text(550, 20, title, 13, "middle", "bold")
    for p, lab in enumerate(labels):
        d = top[top.direction == lab].copy()
        d = d.reindex(d.coef.abs().sort_values(ascending=False).index).head(k)
        _barh_panel(s, d.feature.astype(str).tolist(), d.coef.tolist(), [CLASS_COLORS.get(lab, "grey")] * len(d),
                    200 + p * 550, 55, 300, row, title=f"toward {lab}")
    s.save(path)


def ablation_plot(abl: pd.DataFrame, path: Path):
    d = abl.sort_values("f1_drop_mean", ascending=False)
    row = 28
    s = Svg(620, 110 + len(d) * row)
    s.text(310, 20, "Feature-group ablation (LR, handcrafted, writer-separated)", 13, "middle", "bold")
    _barh_panel(s, d.group.tolist(), d.f1_drop_mean.tolist(), ["#6c5ce7"] * len(d), 150, 45, 420, row,
                errors=d.f1_drop_std.to_numpy())
    s.text(360, 75 + len(d) * row, "drop in macro-F1 when the group is removed (mean ± std over seeds)", 10, "middle")
    s.save(path)


def distribution_plots(features: pd.DataFrame, labels: pd.Series, columns, path: Path):
    cols = [c for c in columns if c in features.columns]
    ncols, pw, ph = 3, 300, 200
    nrows = int(np.ceil(len(cols) / ncols))
    s = Svg(ncols * (pw + 40) + 20, nrows * (ph + 60) + 60)
    s.text(s.w / 2, 22, "Feature distributions by class (full processed dataset)", 13, "middle", "bold")
    for lab_i, lab in enumerate(LABELS):
        s.rect(20 + lab_i * 110, 34, 12, 12, CLASS_COLORS[lab], 0.6)
        s.text(36 + lab_i * 110, 44, lab, 10)
    for i, c in enumerate(cols):
        x0 = 40 + (i % ncols) * (pw + 40)
        y0 = 70 + (i // ncols) * (ph + 60)
        v_all = features[c].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(v_all)), float(np.nanmax(v_all))
        if hi == lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 21)
        hists = {lab: np.histogram(features.loc[labels.values == lab, c], bins=bins, density=True)[0]
                 for lab in LABELS}
        top = max(h.max() for h in hists.values()) or 1
        s.text(x0 + pw / 2, y0 - 6, c, 11, "middle", "bold")
        s.rect(x0, y0, pw, ph, "none", stroke="#999")
        bw = pw / (len(bins) - 1)
        for lab, h in hists.items():
            for j, val in enumerate(h):
                s.rect(x0 + j * bw, y0 + ph - val / top * ph, bw, val / top * ph, CLASS_COLORS[lab], 0.45)
        s.text(x0, y0 + ph + 14, f"{lo:.2f}", 9, "start")
        s.text(x0 + pw, y0 + ph + 14, f"{hi:.2f}", 9, "end")
    s.save(path)
