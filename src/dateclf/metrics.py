from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def precision_recall_at_k(
    y_true: pd.Series,
    scores: pd.Series,
    ks: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    Compute Precision@K and Recall@K on a single ranked list.

    This is used for site-level evaluation:
    all nodes from the held-out site are ranked together by score.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length.")

    if not ks:
        raise ValueError("ks must be a non-empty list of positive integers.")

    if any(k <= 0 for k in ks):
        raise ValueError(f"All K values must be >= 1. Got: {ks}")

    y = y_true.astype(int).to_numpy()
    s = scores.astype(float).to_numpy()

    order = np.argsort(-s)
    total_pos = int(y.sum())

    out: Dict[int, Dict[str, float]] = {}

    for k in ks:
        topk = order[: min(k, len(order))]
        tp = int(y[topk].sum())

        denom_p = len(topk)
        precision = tp / denom_p if denom_p > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else float("nan")

        out[k] = {
            "precision": float(precision),
            "recall": float(recall),
            "num_items": int(len(topk)),
            "num_positives_total": int(total_pos),
        }

    return out