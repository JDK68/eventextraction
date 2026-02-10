from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from dateclf.data import load_config, load_raw_merged, add_is_date_label, build_feature_matrix_for_date
from dateclf.split import loso_folds_date
from dateclf.sampling import undersample_other
from dateclf.metrics import precision_recall_at_k_per_page


@dataclass
class FoldResult:
    site: str
    metrics_at_k: Dict[int, Dict[str, float]]


def train_date_loso(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    ks: List[int] = cfg["eval"]["ks"]

    df = load_raw_merged(cfg)
    df = add_is_date_label(df, cfg)

    X, y, groups = build_feature_matrix_for_date(df)

    # Prepare LightGBM categorical features
    cat_features = [c for c in ["tag", "parent_tag"] if c in X.columns]
    for c in cat_features:
        X[c] = X[c].astype("category")

    folds = list(loso_folds_date(df))

    all_fold_results: List[FoldResult] = []

    fi_all = []

    for i, fold in enumerate(folds, 1):
        train_idx = fold.train_idx
        test_idx = fold.test_idx

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        # undersample negatives in train only
        X_train_s, y_train_s = undersample_other(
            X_train, y_train, seed=cfg["seed"] + i, keep_neg_ratio=0.15
        )

        # scale_pos_weight computed on sampled train
        n_pos = int(y_train_s.sum())
        n_neg = int((y_train_s == 0).sum())
        spw = (n_neg / max(n_pos, 1))

        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "metric": "None",
            "verbosity": -1,
            "seed": cfg["seed"],
            "scale_pos_weight": spw,
        }

        dtrain = lgb.Dataset(
            X_train_s,
            label=y_train_s,
            categorical_feature=cat_features if cat_features else "auto",
            free_raw_data=False,
        )

        # Light early stopping using a small internal split from train
        # (still within train sites only)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=400,
        )

        # --- Feature importance (gain) ---
        fi = pd.DataFrame({
            "feature": model.feature_name(),
            "importance_gain": model.feature_importance(importance_type="gain"),
            "importance_split": model.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False)

        fi_all.append(fi.assign(site=fold.holdout_site))

        # Print top 10
        print("  Top-10 features (gain):")
        print(fi.head(10).to_string(index=False))

        # Predict probabilities on test
        test_scores = model.predict(X_test)

        # Build eval DF with page_id
        df_test = df.loc[test_idx, ["text_context", "is_Date"]].copy()
        df_test = df_test.rename(columns={"text_context": "page_id"})
        df_test["score"] = test_scores

        metrics_at_k = precision_recall_at_k_per_page(
            df_test,
            score_col="score",
            y_col="is_Date",
            page_col="page_id",
            ks=ks,
            ignore_pages_with_no_positives_for_recall=True,
            ignore_pages_with_no_positives_for_precision=False,
        )

        all_fold_results.append(FoldResult(site=fold.holdout_site, metrics_at_k=metrics_at_k))

        print(f"\nFold {i}/{len(folds)} holdout={fold.holdout_site}")
        for k in ks:
            m = metrics_at_k[k]
            print(f"  P@{k}={m['precision']:.3f}  R@{k}={m['recall']:.3f}  "
                  f"(pages P:{m['num_pages_precision']} R:{m['num_pages_recall']})")

        # Save artifacts
        out_dir = Path("runs/date") / fold.holdout_site
        out_dir.mkdir(parents=True, exist_ok=True)

        model.save_model(str(out_dir / "model.txt"))
        (out_dir / "metrics.json").write_text(
            json.dumps({"holdout_site": fold.holdout_site, "metrics_at_k": metrics_at_k}, indent=2),
            encoding="utf-8",
        )
        df_test.to_parquet(out_dir / "preds.parquet", index=False)
        fi.to_csv(out_dir / "feature_importance.csv", index=False)
   
    # --- Aggregate feature importance across folds ---
    fi_cat = pd.concat(fi_all, ignore_index=True)
    fi_mean = (
        fi_cat.groupby("feature")[["importance_gain", "importance_split"]]
        .mean()
        .sort_values("importance_gain", ascending=False)
        .reset_index()
    )
    fi_mean.to_csv("runs/date/feature_importance_mean.csv", index=False)
    print("\n=== Mean Feature Importance across folds (Top 15, gain) ===")
    print(fi_mean.head(15).to_string(index=False))

    # Aggregate macro over folds (average of site means)
    agg = {}
    for k in ks:
        ps = [fr.metrics_at_k[k]["precision"] for fr in all_fold_results]
        rs = [fr.metrics_at_k[k]["recall"] for fr in all_fold_results]
        agg[k] = {"precision_mean": float(np.mean(ps)), "recall_mean": float(np.mean(rs))}

    summary = {
        "num_folds": len(all_fold_results),
        "folds": [{"site": fr.site, "metrics_at_k": fr.metrics_at_k} for fr in all_fold_results],
        "macro_mean": agg,
    }
    Path("runs/date").mkdir(parents=True, exist_ok=True)
    Path("runs/date/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary