from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml


EVENT_CONTENT_LABELS = {
    "Date",
    "DateTime",
    "StartTime",
    "EndTime",
    "StartEndTime",
    "Time",
    "TimeLocation",
    "Location",
    "Name",
    "NameLocation",
    "Description",
}

EVENT_ANCHOR_LABELS = {
    "Date",
    "DateTime",
    "StartTime",
    "EndTime",
    "StartEndTime",
    "Time",
    "TimeLocation",
    "Location",
}

EVENT_FEATURE_COLUMNS = [
    # lightweight structure
    "depth",
    "tag",
    "parent_tag",

    # text statistics
    "text_length",
    "word_count",
    "letter_ratio",
    "digit_ratio",
    "whitespace_ratio",

    # semantic hints
    "contains_date",
    "contains_time",
    "starts_with_digit",
    "ends_with_digit",

    # attribute hints
    "attribute_count",
    "has_class",
    "has_id",

    # keyword features
    "attr_has_word_date",
    "attr_has_word_time",
    "attr_has_word_location",
    "text_has_word_date",
    "text_word_time",
    "text_word_location",

    "siblings_with_date",
    "siblings_with_time",
    "siblings_with_text",
    "num_siblings",
]


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

    sep = dcfg.get("sep")
    if not sep or sep == "TO_FILL":
        raise ValueError("Please set data.sep in config.yaml.")

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

    merged = pd.concat(dfs, ignore_index=True)
    merged["site_id"] = merged["site_id"].astype(str)
    return merged

def add_is_event_content_label(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Binary target for event-content node detection.
    A node is positive if its label belongs to EVENT_CONTENT_LABELS.
    """
    label_col = config["data"]["label_col"]
    if label_col not in df.columns:
        raise ValueError(
            f"label_col='{label_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    labels = df[label_col].fillna("").astype(str)

    positive_labels = config.get("targets", {}).get(
        "event_content_positive_labels",
        sorted(EVENT_CONTENT_LABELS),
    )
    if not positive_labels:
        raise ValueError("targets.event_content_positive_labels is empty.")

    out = df.copy()
    out["is_event_content"] = labels.isin(positive_labels).astype(int)
    return out
    

def build_feature_matrix_for_event(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Feature matrix for event-content node detection.
    Returns X, y, groups.
    """
    if "is_event_content" not in df.columns:
        raise ValueError(
            "is_event_content missing. Call add_is_event_content_label() first."
        )
    if "site_id" not in df.columns:
        raise ValueError("site_id missing. Ensure loader adds site_id.")

    feature_cols = [c for c in EVENT_FEATURE_COLUMNS if c in df.columns]

    X = df[feature_cols].copy()
    y = df["is_event_content"].astype(int)
    groups = df["site_id"].astype(str)

    for c in ["tag", "parent_tag"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X, y, groups

def add_is_event_anchor_label(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Binary target for anchor-like event nodes.
    A node is positive if its label belongs to the configured
    event-anchor positive labels.
    """
    label_col = config["data"]["label_col"]
    if label_col not in df.columns:
        raise ValueError(
            f"label_col='{label_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    labels = df[label_col].fillna("").astype(str)

    positive_labels = config.get("targets", {}).get(
        "event_anchor_positive_labels",
        sorted(EVENT_ANCHOR_LABELS),
    )
    if not positive_labels:
        raise ValueError("targets.event_anchor_positive_labels is empty.")

    out = df.copy()
    out["is_event_anchor"] = labels.isin(positive_labels).astype(int)
    return out

def build_feature_matrix_for_anchor(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Feature matrix for event-anchor node detection.
    Returns X, y, groups.
    """
    if "is_event_anchor" not in df.columns:
        raise ValueError(
            "is_event_anchor missing. Call add_is_event_anchor_label() first."
        )
    if "site_id" not in df.columns:
        raise ValueError("site_id missing. Ensure loader adds site_id.")

    feature_cols = [c for c in EVENT_FEATURE_COLUMNS if c in df.columns]

    X = df[feature_cols].copy()
    y = df["is_event_anchor"].astype(int)
    groups = df["site_id"].astype(str)

    for c in ["tag", "parent_tag"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X, y, groups