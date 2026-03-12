from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from dateclf.data import (
    load_config,
    load_raw_merged,
    add_is_event_anchor_label,
    build_feature_matrix_for_anchor,
)
from dateclf.split import loso_folds_event
from dateclf.sampling import undersample_negatives
from dateclf.metrics import precision_recall_at_k


@dataclass
class FoldResult:
    site: str
    metrics_at_k: Dict[int, Dict[str, float]]


def _build_lgb_params(cfg: Dict[str, Any], scale_pos_weight: float) -> Dict[str, Any]:
    model_cfg = cfg.get("model", {})

    return {
        "objective": "binary",
        "learning_rate": model_cfg.get("learning_rate", 0.05),
        "num_leaves": model_cfg.get("num_leaves", 63),
        "min_data_in_leaf": model_cfg.get("min_data_in_leaf", 30),
        "feature_fraction": model_cfg.get("feature_fraction", 0.9),
        "bagging_fraction": model_cfg.get("bagging_fraction", 0.8),
        "bagging_freq": model_cfg.get("bagging_freq", 1),
        "lambda_l2": model_cfg.get("lambda_l2", 1.0),
        "metric": "None",
        "verbosity": -1,
        "seed": cfg["seed"],
        "scale_pos_weight": scale_pos_weight,
    }


def _save_fold_artifacts(
    out_dir: Path,
    holdout_site: str,
    model: lgb.Booster,
    metrics_at_k: Dict[int, Dict[str, float]],
    df_test: pd.DataFrame,
    fi: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(out_dir / "model.txt"))
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "holdout_site": holdout_site,
                "metrics_at_k": metrics_at_k,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    df_test.to_csv(out_dir / "preds.csv", index=False)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)


def train_event_anchor_loso(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    ks: List[int] = cfg["eval"]["ks"]
    keep_neg_ratio = cfg.get("sampling", {}).get("keep_neg_ratio", 0.15)
    num_boost_round = cfg.get("model", {}).get("num_boost_round", 400)

    df = load_raw_merged(cfg)
    
    from .features import add_dom_neighbor_features

    df = add_dom_neighbor_features(df)

    df = add_is_event_anchor_label(df, cfg)
    X, y, groups = build_feature_matrix_for_anchor(df)

    cat_features = [c for c in ["tag", "parent_tag"] if c in X.columns]

    folds = list(loso_folds_event(df, y_col="is_event_anchor"))

    all_fold_results: List[FoldResult] = []
    fi_all: List[pd.DataFrame] = []

    for i, fold in enumerate(folds, start=1):
        train_idx = fold.train_idx
        test_idx = fold.test_idx

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        X_train_s, y_train_s = undersample_negatives(
            X_train,
            y_train,
            seed=cfg["seed"] + i,
            keep_neg_ratio=keep_neg_ratio,
        )

        n_pos = int(y_train_s.sum())
        n_neg = int((y_train_s == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        params = _build_lgb_params(cfg, scale_pos_weight)

        dtrain = lgb.Dataset(
            X_train_s,
            label=y_train_s,
            categorical_feature=cat_features if cat_features else "auto",
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
        )

        fi = pd.DataFrame(
            {
                "feature": model.feature_name(),
                "importance_gain": model.feature_importance(importance_type="gain"),
                "importance_split": model.feature_importance(importance_type="split"),
            }
        ).sort_values("importance_gain", ascending=False)

        fi_all.append(fi.assign(site=fold.holdout_site))

        print(f"\nFold {i}/{len(folds)} holdout={fold.holdout_site}")
        print("  Top-10 features (gain):")
        print(fi.head(10).to_string(index=False))

        test_scores = model.predict(X_test)

        df_test = df.loc[test_idx, ["text_context", "label", "event_id", "is_event_anchor"]].copy()
        df_test["score"] = test_scores

        metrics_at_k = precision_recall_at_k(
            y_true=df_test["is_event_anchor"],
            scores=df_test["score"],
            ks=ks,
        )

        all_fold_results.append(
            FoldResult(site=fold.holdout_site, metrics_at_k=metrics_at_k)
        )

        for k in ks:
            m = metrics_at_k[k]
            print(
                f"  P@{k}={m['precision']:.3f}  "
                f"R@{k}={m['recall']:.3f}  "
                f"(top_k:{m['num_items']} total_pos:{m['num_positives_total']})"
            )

        out_dir = Path("runs/anchor") / fold.holdout_site
        _save_fold_artifacts(
            out_dir=out_dir,
            holdout_site=fold.holdout_site,
            model=model,
            metrics_at_k=metrics_at_k,
            df_test=df_test,
            fi=fi,
        )

    fi_cat = pd.concat(fi_all, ignore_index=True)
    fi_mean = (
        fi_cat.groupby("feature")[["importance_gain", "importance_split"]]
        .mean()
        .sort_values("importance_gain", ascending=False)
        .reset_index()
    )

    Path("runs/anchor").mkdir(parents=True, exist_ok=True)
    fi_mean.to_csv("runs/anchor/feature_importance_mean.csv", index=False)

    print("\n=== Mean Feature Importance across folds (Top 15, gain) ===")
    print(fi_mean.head(15).to_string(index=False))

    agg = {}
    for k in ks:
        ps = [fr.metrics_at_k[k]["precision"] for fr in all_fold_results]
        rs = [fr.metrics_at_k[k]["recall"] for fr in all_fold_results]
        agg[k] = {
            "precision_mean": float(np.mean(ps)),
            "recall_mean": float(np.mean(rs)),
        }

    summary = {
        "num_folds": len(all_fold_results),
        "folds": [
            {"site": fr.site, "metrics_at_k": fr.metrics_at_k}
            for fr in all_fold_results
        ],
        "macro_mean": agg,
    }

    Path("runs/anchor/summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


if __name__ == "__main__":
    summary = train_event_anchor_loso("../config.yaml")
    print("\nTraining finished.")
    print(summary["macro_mean"])