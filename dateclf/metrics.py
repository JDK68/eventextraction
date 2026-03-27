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

def event_level_metrics_at_k(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    event_id_col: str,
    ks: list[int],
) -> dict[int, dict[str, float]]:
    out = {}

    ranked = df.sort_values(score_col, ascending=False).copy()
    events_df = df[df[event_id_col].notna()].copy()

    grouped_all = events_df.groupby(event_id_col)

    for k in ks:
        topk = ranked.head(k).copy()
        topk_events = topk[topk[event_id_col].notna()].copy()
        grouped_topk = topk_events.groupby(event_id_col) if len(topk_events) > 0 else None

        recalls = []
        detected = []

        for event_id, group in grouped_all:
            total_pos = int(group[y_col].sum())
            if total_pos == 0:
                continue

            captured = 0
            if grouped_topk is not None and event_id in grouped_topk.groups:
                top_group = grouped_topk.get_group(event_id)
                captured = int(top_group[y_col].sum())

            recalls.append(captured / total_pos)
            detected.append(1.0 if captured > 0 else 0.0)

        out[k] = {
            "event_detection_rate": float(sum(detected) / len(detected)) if detected else 0.0,
            "mean_event_recall": float(sum(recalls) / len(recalls)) if recalls else 0.0,
            "num_events": int(len(recalls)),
        }

    return out