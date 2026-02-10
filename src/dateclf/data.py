from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_site_id_from_filename(path: Path) -> str:
    return path.stem


def load_raw_merged(config: Dict[str, Any]) -> pd.DataFrame:
    dcfg = config["data"]
    raw_dir = Path(dcfg["raw_dir"])
    files = sorted(raw_dir.glob(dcfg.get("file_glob", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir.resolve()}")

    sep = dcfg.get("sep", None)
    if not sep or sep == "TO_FILL":
        raise ValueError("Please set data.sep in config.yaml (your CSV separator).")

    encoding = dcfg.get("encoding", "utf-8")
    site_source = dcfg.get("site_id_source", "filename")

    dfs: List[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp, sep=sep, encoding=encoding, engine="python")

        if site_source == "filename":
            df["site_id"] = _infer_site_id_from_filename(fp)
        elif site_source == "column":
            if "site_id" not in df.columns:
                raise ValueError(f"'site_id' column missing in {fp.name}")
            df["site_id"] = df["site_id"].astype(str)
        else:
            raise ValueError("site_id_source must be 'filename' or 'column'")

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["site_id"] = out["site_id"].astype(str)
    return out


def add_is_date_label(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create binary target is_Date.
    Rule: is_Date = 1 iff label contains 'date' (case-insensitive).
    """
    label_col = config["data"]["label_col"]
    if label_col not in df.columns:
        raise ValueError(
            f"label_col='{label_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    labels = df[label_col].astype(str).fillna("").str.lower()

    df = df.copy()
    df["is_Date"] = labels.str.contains("date").astype(int)

    return df

def build_feature_matrix_for_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build X, y, groups for Date classifier.
    - X: features only (no leakage columns)
    - y: binary target is_Date
    - groups: site_id (for LOSO splitting, not a feature)
    """
    exclude_cols = {
        # leakage / targets / grouping
        "label",
        "is_Date",
        "site_id",
        "event_id",

        # raw strings that are site-specific or too free-form for Phase 2 baseline
        "attributes",
        "text_context",
        "link",  # often raw URL/path; can be very site-specific
    }

    cols_to_drop = [c for c in exclude_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Keep y + groups separately
    if "is_Date" not in df.columns:
        raise ValueError("is_Date missing. Call add_is_date_label() first.")
    y = df["is_Date"].astype(int)

    if "site_id" not in df.columns:
        raise ValueError("site_id missing. Ensure loader adds site_id.")
    groups = df["site_id"].astype(str)

        # Treat tag fields as categorical (LightGBM can use them directly)
    for c in ["tag", "parent_tag"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X, y, groups