import numpy as np
import pandas as pd


def add_dom_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add DOM neighbourhood and local rendering-context features.
    Assumes one page/site per CSV and that rendering_order is meaningful within that page.
    """
    df = df.copy()

    # ---------- Parent/sibling-based features ----------
    parent_group = df.groupby("parent_index", dropna=False)

    # Number of siblings with date/time-like signals
    df["siblings_with_date"] = (
        parent_group["contains_date"].transform("sum") - df["contains_date"]
    )
    df["siblings_with_time"] = (
        parent_group["contains_time"].transform("sum") - df["contains_time"]
    )

    # Number of siblings with long text (often description-like)
    df["siblings_with_text"] = (
        parent_group["text_length"].transform(lambda x: (x > 40).sum())
        - (df["text_length"] > 40).astype(int)
    )

    # Number of siblings
    df["num_siblings"] = parent_group["parent_index"].transform("count") - 1

    # Parent-level aggregate features
    df["parent_contains_date_count"] = parent_group["contains_date"].transform("sum")
    df["parent_contains_time_count"] = parent_group["contains_time"].transform("sum")
    df["parent_long_text_count"] = parent_group["text_length"].transform(
        lambda x: (x > 40).sum()
    )
    df["parent_avg_text_length"] = parent_group["text_length"].transform("mean").fillna(0)

    # Position among siblings based on rendering_order
    df["_sibling_rank"] = parent_group["rendering_order"].rank(method="first") - 1
    df["_sibling_count"] = parent_group["rendering_order"].transform("count")

    df["sibling_position"] = df["_sibling_rank"].fillna(0).astype(float)
    df["sibling_position_norm"] = np.where(
        df["_sibling_count"] > 1,
        df["_sibling_rank"] / (df["_sibling_count"] - 1),
        0.0,
    )
    df["is_first_sibling"] = (df["_sibling_rank"] == 0).astype(int)
    df["is_last_sibling"] = (df["_sibling_rank"] == (df["_sibling_count"] - 1)).astype(int)

    # ---------- Local rendering-order neighbour features ----------
    # Sort by rendering order to compute prev/next local context
    order_df = df.sort_values("rendering_order").copy()

    order_df["prev_text_length"] = order_df["text_length"].shift(1).fillna(0)
    order_df["next_text_length"] = order_df["text_length"].shift(-1).fillna(0)

    order_df["prev_contains_date"] = (
        pd.to_numeric(order_df["contains_date"].shift(1), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    order_df["next_contains_date"] = (
        pd.to_numeric(order_df["contains_date"].shift(-1), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    order_df["prev_contains_time"] = (
        pd.to_numeric(order_df["contains_time"].shift(1), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    order_df["next_contains_time"] = (
        pd.to_numeric(order_df["contains_time"].shift(-1), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    order_df["prev_parent_index"] = order_df["parent_index"].shift(1)
    order_df["next_parent_index"] = order_df["parent_index"].shift(-1)

    order_df["same_parent_as_prev"] = (
        order_df["parent_index"].notna()
        & order_df["prev_parent_index"].notna()
        & (order_df["parent_index"] == order_df["prev_parent_index"])
    ).astype(int)

    order_df["same_parent_as_next"] = (
        order_df["parent_index"].notna()
        & order_df["next_parent_index"].notna()
        & (order_df["parent_index"] == order_df["next_parent_index"])
    ).astype(int)

    order_df["prev_rendering_order"] = order_df["rendering_order"].shift(1)
    order_df["next_rendering_order"] = order_df["rendering_order"].shift(-1)

    order_df["rendering_gap_prev"] = (
        order_df["rendering_order"] - order_df["prev_rendering_order"]
    ).fillna(0)

    order_df["rendering_gap_next"] = (
        order_df["next_rendering_order"] - order_df["rendering_order"]
    ).fillna(0)

    # Merge back in original row order
    cols_to_copy = [
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
    ]
    df = df.join(order_df[cols_to_copy])

    # Cleanup temp cols
    df = df.drop(columns=["_sibling_rank", "_sibling_count"], errors="ignore")

    # Fill any remaining NaNs from edge cases
    fill_zero_cols = [
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
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    int_cols = [
        "siblings_with_date",
        "siblings_with_time",
        "siblings_with_text",
        "num_siblings",
        "parent_contains_date_count",
        "parent_contains_time_count",
        "parent_long_text_count",
        "is_first_sibling",
        "is_last_sibling",
        "prev_contains_date",
        "next_contains_date",
        "prev_contains_time",
        "next_contains_time",
        "same_parent_as_prev",
        "same_parent_as_next",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    float_cols = [
        "parent_avg_text_length",
        "sibling_position",
        "sibling_position_norm",
        "prev_text_length",
        "next_text_length",
        "rendering_gap_prev",
        "rendering_gap_next",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    return df

def add_event_density_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    parent_group = df.groupby("parent_index")

    # combien de signaux event autour
    df["parent_date_count"] = parent_group["contains_date"].transform("sum")
    df["parent_time_count"] = parent_group["contains_time"].transform("sum")
    df["parent_text_rich_count"] = parent_group["text_length"].transform(
        lambda x: (x > 30).sum()
    )

    # signal combiné
    df["parent_event_density"] = (
        df["parent_date_count"]
        + df["parent_time_count"]
        + df["parent_text_rich_count"]
    )

    return df