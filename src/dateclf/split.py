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
    y_col: str = "is_event_content"
) -> Iterator[Fold]:
    """
    Leave-One-Site-Out (LOSO) folds for event detection.

    We exclude sites that contain zero positive event nodes in the test site
    to avoid undefined Recall@K and misleading evaluation.
    """

    # Count positives per site
    pos_by_site = df.groupby(site_col)[y_col].sum()

    # Only keep sites that have at least one positive event node
    eligible_sites: List[str] = sorted(
        pos_by_site[pos_by_site > 0].index.astype(str).tolist()
    )

    for s in eligible_sites:
        test_mask = df[site_col].astype(str) == str(s)

        yield Fold(
            holdout_site=str(s),
            train_idx=df.index[~test_mask],
            test_idx=df.index[test_mask],
        )