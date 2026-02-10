from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def precision_recall_at_k_per_page(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    page_col: str,
    ks: List[int],
    ignore_pages_with_no_positives_for_recall: bool = True,
    ignore_pages_with_no_positives_for_precision: bool = False,
) -> Dict[int, Dict[str, float]]:
    """
    Compute macro-averaged Precision@K and Recall@K across pages.
    For each page: sort nodes by score desc, take top-K.
    """
    out: Dict[int, Dict[str, float]] = {}

    pages = df[page_col].unique().tolist()
    for k in ks:
        precisions = []
        recalls = []

        for pid in pages:
            g = df[df[page_col] == pid]
            y = g[y_col].astype(int).to_numpy()
            scores = g[score_col].astype(float).to_numpy()

            # sort desc
            order = np.argsort(-scores)
            topk = order[: min(k, len(order))]

            tp = int(y[topk].sum())
            denom_p = min(k, len(order))
            p_at_k = tp / denom_p if denom_p > 0 else 0.0

            total_pos = int(y.sum())
            if total_pos == 0:
                if not ignore_pages_with_no_positives_for_precision:
                    precisions.append(p_at_k)
                if not ignore_pages_with_no_positives_for_recall:
                    recalls.append(0.0)
                continue

            r_at_k = tp / total_pos

            precisions.append(p_at_k)
            recalls.append(r_at_k)

        out[k] = {
            "precision": float(np.mean(precisions)) if precisions else float("nan"),
            "recall": float(np.mean(recalls)) if recalls else float("nan"),
            "num_pages_precision": int(len(precisions)),
            "num_pages_recall": int(len(recalls)),
        }

    return out