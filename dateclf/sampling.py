from __future__ import annotations

import numpy as np
import pandas as pd


def undersample_negatives(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    keep_neg_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Randomly undersample the negative class.

    Keeps all positive examples and retains only a fraction of negatives.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index.")

    if not (0 < keep_neg_ratio <= 1):
        raise ValueError(
            f"keep_neg_ratio must be in (0, 1], got {keep_neg_ratio}"
        )

    rng = np.random.default_rng(seed)

    y = y.astype(int)
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_keep_neg = int(len(neg_idx) * keep_neg_ratio)
    n_keep_neg = max(1, min(len(neg_idx), n_keep_neg))

    keep_neg = rng.choice(neg_idx.to_numpy(), size=n_keep_neg, replace=False)

    keep_idx = pos_idx.append(pd.Index(keep_neg)).unique()
    keep_idx = keep_idx.sort_values()

    return X.loc[keep_idx], y.loc[keep_idx]