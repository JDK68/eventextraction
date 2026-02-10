from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List
import pandas as pd


@dataclass(frozen=True)
class Fold:
    holdout_site: str
    train_idx: pd.Index
    test_idx: pd.Index


def loso_folds_date(df: pd.DataFrame, site_col: str = "site_id", y_col: str = "is_Date") -> Iterator[Fold]:
    """
    LOSO folds, excluding sites that have zero positives in the test site.
    This prevents undefined Recall@K and misleading evaluation.
    """
    # sites with at least 1 positive
    pos_by_site = df.groupby(site_col)[y_col].sum()
    eligible_sites: List[str] = sorted(pos_by_site[pos_by_site > 0].index.astype(str).tolist())

    for s in eligible_sites:
        test_mask = df[site_col].astype(str) == str(s)
        yield Fold(
            holdout_site=str(s),
            train_idx=df.index[~test_mask],
            test_idx=df.index[test_mask],
        )