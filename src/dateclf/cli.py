from __future__ import annotations
import argparse
from dateclf.data import load_config, load_raw_merged, add_is_date_label, build_feature_matrix_for_date
from dateclf.split import loso_folds_date
from dateclf.train import train_date_loso

def cmd_data_check(config_path: str) -> int:
    cfg = load_config(config_path)
    df = load_raw_merged(cfg)

    print("OK: merged CSVs")
    print("rows:", len(df))
    print("sites:", df["site_id"].nunique())
    print("columns:", len(df.columns))
    print("first cols:", df.columns.tolist()[:20])

    label_col = cfg["data"]["label_col"]
    if label_col in df.columns:
        vc = df[label_col].astype(str).value_counts().head(20)
        print("\nTop label values:")
        print(vc.to_string())

    df = add_is_date_label(df, cfg)

    print("\nTarget is_Date stats:")
    print("positives:", int(df["is_Date"].sum()))
    print("positive rate:", round(float(df["is_Date"].mean()), 4))

    print("\nPositives per site:")
    site_stats = (
        df.groupby("site_id")["is_Date"]
        .agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False)
    )
    print(site_stats.to_string())

    return 0

def cmd_feature_check(config_path: str) -> int:
    cfg = load_config(config_path)
    df = load_raw_merged(cfg)
    df = add_is_date_label(df, cfg)

    X, y, groups = build_feature_matrix_for_date(df)

    print("OK: feature matrix built (Date)")
    print("X shape:", X.shape)
    print("y positives:", int(y.sum()), "rate:", round(float(y.mean()), 4))
    print("num sites (groups):", groups.nunique())

    # Types summary
    print("\nDtypes count:")
    print(X.dtypes.value_counts().to_string())

    # Identify non-numeric columns (LightGBM can handle categoricals, but we'll decide explicitly)
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print("\nObject columns:", obj_cols)

    # NaN check
    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if len(nan_cols) == 0:
        print("\nNaN check: OK (no missing values)")
    else:
        print("\nNaN columns (top):")
        print(nan_cols.head(20).to_string())
        # % missing for worst offenders
        worst = nan_cols.head(10)
        print("\nNaN rate (top):")
        print(((worst / len(X)).round(4)).to_string())

    # Leakage assertions
    assert "label" not in X.columns
    assert "site_id" not in X.columns
    assert "is_Date" not in X.columns
    assert "event_id" not in X.columns

    return 0

def cmd_fold_check(config_path: str) -> int:
    cfg = load_config(config_path)
    df = load_raw_merged(cfg)
    df = add_is_date_label(df, cfg)

    folds = list(loso_folds_date(df))
    print("Eligible LOSO folds:", len(folds))
    print("Holdout sites:")
    for f in folds:
        print(" -", f.holdout_site)

    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("command", choices=["data-check", "feature-check", "fold-check", "train-date"])
    args = p.parse_args(argv)

    if args.command == "data-check":
        return cmd_data_check(args.config)
    if args.command == "feature-check":
        return cmd_feature_check(args.config)
    if args.command == "fold-check":
        return cmd_fold_check(args.config)
    if args.command == "train-date":
        train_date_loso(args.config)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())