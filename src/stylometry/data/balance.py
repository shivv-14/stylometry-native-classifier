"""Writer-level class balancing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _allocate(sizes: pd.Series, total: int) -> pd.Series:
    """Proportional allocation of `total` across strata (largest remainder)."""
    raw = sizes / sizes.sum() * total
    alloc = np.floor(raw).astype(int)
    alloc = alloc.clip(upper=sizes)
    rest = total - alloc.sum()
    order = (raw - alloc).sort_values(ascending=False).index
    for key in order:
        if rest <= 0:
            break
        if alloc[key] < sizes[key]:
            alloc[key] += 1
            rest -= 1
    return alloc


def balance_writers(df: pd.DataFrame, seed: int, strata: list[str] | None = None,
                    majority: str | None = None) -> pd.DataFrame:
    """Downsample writers of the larger class to the number of writers in the smaller class.

    Sampling is stratified by the given metadata columns (e.g. l1, proficiency) when
    they exist. All texts of each selected writer are kept.
    """
    writers = df.groupby("writer_id").first().reset_index()
    counts = writers["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need both classes to balance")
    minority = counts.idxmin()
    majority = majority or next(lab for lab in counts.index if lab != minority)
    target = int(counts[minority])
    if counts[majority] <= target:          # already balanced
        return df.reset_index(drop=True)
    maj = writers[writers.label == majority]

    strata = [c for c in (strata or []) if c in maj.columns and maj[c].notna().any()]
    rng = np.random.default_rng(seed)
    if strata:
        key = maj[strata].astype(str).agg("|".join, axis=1)
        sizes = key.value_counts()
        alloc = _allocate(sizes, target)
        chosen = []
        for k, n in alloc.items():
            ids = maj.loc[key == k, "writer_id"].to_numpy()
            chosen.extend(rng.choice(ids, size=int(n), replace=False))
    else:
        chosen = rng.choice(maj["writer_id"].to_numpy(), size=target, replace=False)

    keep = set(writers.loc[writers.label == minority, "writer_id"]) | set(chosen)
    return df[df.writer_id.isin(keep)].reset_index(drop=True)
