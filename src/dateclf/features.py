import pandas as pd


def add_dom_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add DOM neighbourhood features using parent_index.
    """

    df = df.copy()

    # number of nodes sharing the same parent
    parent_group = df.groupby("parent_index")

    # siblings with date
    df["siblings_with_date"] = parent_group["contains_date"].transform("sum") - df["contains_date"]

    # siblings with time
    df["siblings_with_time"] = parent_group["contains_time"].transform("sum") - df["contains_time"]

    # siblings with long text (often description)
    df["siblings_with_text"] = parent_group["text_length"].transform(
        lambda x: (x > 40).sum()
    )

    # siblings count
    df["num_siblings"] = parent_group["parent_index"].transform("count") - 1

    return df