from __future__ import annotations
import numpy as np
import pandas as pd


def undersample_other(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    keep_neg_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Keep all positives.
    Keep a fraction of negatives (Other) to reduce imbalance.
    """
    rng = np.random.default_rng(seed)

    y = y.astype(int)
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_keep_neg = int(len(neg_idx) * keep_neg_ratio)
    if n_keep_neg < 1:
        n_keep_neg = min(len(neg_idx), 1)

    keep_neg = rng.choice(neg_idx.to_numpy(), size=n_keep_neg, replace=False)
    keep_idx = pos_idx.union(pd.Index(keep_neg))

    return X.loc[keep_idx], y.loc[keep_idx]