"""Run all experiments (B0, B1, E1-E6) and write results + figures.

    python scripts/run_experiments.py [config.yaml]

Outputs
  reports/results/runs.csv       one row per experiment x seed
  reports/results/results.csv    mean / std over seeds
  reports/results/ablation.csv   E5 drop in macro-F1 per feature group
  reports/results/l1_recall.csv  recall per L1 for the best model (if L1 metadata exists)
  reports/results/group_importance.csv  sum |coef| per handcrafted group
  reports/figures/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stylometry.config import load_config, rel, resolve, set_seed  # noqa: E402
from stylometry.evaluation import plots  # noqa: E402
from stylometry.evaluation.evaluate import aggregate, run_seeds  # noqa: E402
from stylometry.explain.explain import coef_group_importance, top_features  # noqa: E402
from stylometry.features.handcrafted import GROUPS, load_disk_cache, save_disk_cache  # noqa: E402
from stylometry.models.pipelines import build_features  # noqa: E402

MAIN_SETS = ("handcrafted", "char", "combined")
MAIN_MODELS = ("lr", "svm")


def main(config_path: str | None = None) -> int:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    data_path = resolve(cfg, "processed")
    if not data_path.exists():
        print(f"{rel(data_path)} not found. Run `python tasks.py prepare` first.")
        return 0
    df = pd.read_parquet(data_path)
    seeds = cfg["seeds"]
    res_dir, fig_dir = resolve(cfg, "results_dir"), resolve(cfg, "figures_dir")
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache = resolve(cfg, "feature_cache")
    load_disk_cache(cache)
    pos = cfg["data"]["positive_label"]
    k = cfg["evaluation"]["top_k_features"]

    all_runs: list = []

    def record(results):
        all_runs.extend(results)
        save_disk_cache(cache)
        return results

    print("baselines")
    record(run_seeds("B0", "none", "dummy", df, cfg, seeds))
    b1 = record(run_seeds("B1", "length", "lr", df, cfg, seeds))

    print("E1-E3: feature sets x models, writer-separated")
    main_runs = {}
    for exp_id, fs in zip(("E1", "E2", "E3"), MAIN_SETS):
        for m in MAIN_MODELS:
            main_runs[(fs, m)] = record(run_seeds(exp_id, fs, m, df, cfg, seeds))

    # best = highest mean CV macro-F1 (training folds only), so the test set does not pick the winner
    cv_mean = {key: np.nanmean([r.row["cv_f1_macro"] for r in rs]) for key, rs in main_runs.items()}
    best_fs, best_model = max(cv_mean, key=cv_mean.get)
    print(f"best by CV: {best_fs} / {best_model}")

    print("E4: document-level random split vs writer-separated")
    for m in MAIN_MODELS:
        record(run_seeds("E4", best_fs, m, df, cfg, seeds, split="document", variant="document-random"))
        for r in main_runs[(best_fs, m)]:
            row = dict(r.row, exp_id="E4", variant="writer-separated")
            all_runs.append(type(r)(row, r.pipeline, r.test, r.y_pred, r.y_score))

    print("E5: handcrafted group ablation (LR)")
    full = main_runs[("handcrafted", "lr")]
    for r in full:
        all_runs.append(type(r)(dict(r.row, exp_id="E5", variant="all groups"), r.pipeline, r.test,
                                r.y_pred, r.y_score))
    ablation_rows = []
    for g in GROUPS:
        keep = [x for x in GROUPS if x != g]
        rs = record(run_seeds("E5", "handcrafted", "lr", df, cfg, seeds, feature_groups=keep,
                              variant=f"without {g}"))
        drops = [f.row["f1_macro"] - r.row["f1_macro"] for f, r in zip(full, rs)]
        ablation_rows.append({"group": g, "f1_drop_mean": np.mean(drops), "f1_drop_std": np.std(drops)})
    ablation = pd.DataFrame(ablation_rows).sort_values("f1_drop_mean", ascending=False)
    ablation.to_csv(res_dir / "ablation.csv", index=False)
    plots.ablation_plot(ablation, fig_dir / "ablation.png")

    topics = sorted(df["topic"].dropna().unique())
    if len(topics) >= 2:
        print("E6: cross-topic")
        for t in topics:
            record(run_seeds("E6", best_fs, best_model, df, cfg, seeds, split="topic", train_topic=t,
                             variant=f"train {t}"))
    else:
        print("E6 skipped: fewer than two topics in the data")

    # ---- tables --------------------------------------------------------------
    runs = pd.DataFrame([r.row for r in all_runs])
    runs.to_csv(res_dir / "runs.csv", index=False)
    results = aggregate(runs)
    results.to_csv(res_dir / "results.csv", index=False)
    pd.Series({"best_feature_set": best_fs, "best_model": best_model}).to_json(res_dir / "best.json")

    # per-L1 recall of the best model, pooled over seeds (fairness check)
    best_runs = main_runs[(best_fs, best_model)]
    pooled = pd.concat([r.test.assign(pred=r.y_pred, seed=r.row["seed"]) for r in best_runs])
    pooled.drop(columns=["text"]).to_csv(res_dir / "predictions_best.csv", index=False)
    pooled = pooled.assign(correct=pooled.pred == pooled.label)
    for col, fname in (("l1", "l1_recall.csv"), ("proficiency", "proficiency_recall.csv")):
        # only useful when the column varies within a class
        if pooled[col].notna().any() and pooled.groupby("label")[col].nunique().max() > 1:
            (pooled.groupby(["label", col]).agg(texts=("correct", "size"), recall=("correct", "mean"))
             .reset_index().to_csv(res_dir / fname, index=False))

    # ---- figures (seed 0 split) ---------------------------------------------
    curves = []
    for (fs, m), rs in main_runs.items():
        r = rs[0]
        plots.confusion_matrix_plot(r.test.label, r.y_pred, f"{fs} / {m} (seed {r.row['seed']})",
                                    fig_dir / f"confusion_{fs}_{m}.png")
        if r.y_score is not None:
            curves.append((f"{fs}/{m}", r.test.label, r.y_score))
        top = top_features(r.pipeline, k)
        if not top.empty:
            top.to_csv(res_dir / f"top_features_{fs}_{m}.csv", index=False)
            plots.top_features_plot(top, f"Top {k} coefficients: {fs} / {m} "
                                    "(associated with the prediction on this dataset)",
                                    fig_dir / f"top_features_{fs}_{m}.png", k)
    if b1[0].y_score is not None:
        curves.append(("length only (B1)", b1[0].test.label, b1[0].y_score))
    plots.roc_plot(curves, fig_dir / "roc_curves.png", pos)

    gi = coef_group_importance(main_runs[("handcrafted", "lr")][0].pipeline)
    gi.to_csv(res_dir / "group_importance.csv", index=False)

    hand = build_features("handcrafted", cfg).named_steps["hand"]
    feats = hand.fit(df.text).transform(df.text)
    plots.distribution_plots(feats, df.label, cfg["evaluation"]["distribution_features"],
                             fig_dir / "feature_distributions.png")
    save_disk_cache(cache)
    print(f"wrote {rel(res_dir)} and {rel(fig_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
