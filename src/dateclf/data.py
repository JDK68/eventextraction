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

    "parent_contains_date_count",
    "parent_contains_time_count",
    "parent_long_text_count",
    "parent_avg_text_length",
    "sibling_position",
    "sibling_position_norm",
    "is_first_sibling",
    "is_last_sibling",
    "prev_text_length",
    "next_text_length",
    "prev_contains_date",
    "next_contains_date",
    "prev_contains_time",
    "next_contains_time",
    "same_parent_as_prev",
    "same_parent_as_next",
    "rendering_gap_prev",
    "rendering_gap_next",

    # NER semantic hints
    "has_ner_date",
    "has_ner_time",
    "has_ner_gpe",
    "has_ner_loc",
    "has_ner_org",
    "has_any_ner",
    "ner_count",
    "has_ner_location_like",
    "has_ner_datetime_like",

    "parent_date_count",
    "parent_time_count",
    "parent_text_rich_count",
    "parent_event_density",

    "is_nav_like",
    "is_contact_like",
    "has_event_keyword",
    "looks_like_location",
    "local_support_count",
    "local_anchor_count",
    "is_meta_noise",
    "looks_like_fragment",
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

def add_is_event_member_label(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Binary target for full event-member detection.
    A node is positive if it belongs to any labeled event.
    This uses event_id directly instead of a manual label list.
    """
    if "event_id" not in df.columns:
        raise ValueError(
            f"'event_id' column not found. Available columns: {df.columns.tolist()}"
        )

    out = df.copy()
    out["is_event_member"] = df["event_id"].notna().astype(int)
    return out


def build_feature_matrix_for_member(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Feature matrix for full event-member node detection.
    Returns X, y, groups.
    """
    if "is_event_member" not in df.columns:
        raise ValueError(
            "is_event_member missing. Call add_is_event_member_label() first."
        )
    if "site_id" not in df.columns:
        raise ValueError("site_id missing. Ensure loader adds site_id.")

    feature_cols = [c for c in EVENT_FEATURE_COLUMNS if c in df.columns]

    X = df[feature_cols].copy()
    y = df["is_event_member"].astype(int)
    groups = df["site_id"].astype(str)

    for c in ["tag", "parent_tag"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    return X, y, groups