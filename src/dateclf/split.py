from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd


@dataclass(frozen=True)
class Fold:
    holdout_site: str
    train_idx: pd.Index
    test_idx: pd.Index


def loso_folds_event(
    df: pd.DataFrame,
    site_col: str = "site_id",
    y_col: str = "is_event_content",
) -> Iterator[Fold]:
    """
    Leave-One-Site-Out (LOSO) folds for event-content node detection.

    Only sites with at least one positive node are used as holdout sites.
    This avoids undefined recall-based metrics on fully negative test sites.
    """
    required_cols = [site_col, y_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for LOSO split: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    site_values = df[site_col].astype(str)

    pos_by_site = df.assign(_site=site_values).groupby("_site")[y_col].sum()

    eligible_sites: List[str] = sorted(
        pos_by_site[pos_by_site > 0].index.tolist()
    )

    for site in eligible_sites:
        test_mask = site_values == site

        yield Fold(
            holdout_site=site,
            train_idx=df.index[~test_mask],
            test_idx=df.index[test_mask],
        )