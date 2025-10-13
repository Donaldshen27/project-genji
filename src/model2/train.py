"""
Model 2 Training Pipeline: Purged & Embargoed TimeSeriesSplit

Implements Chunk 3 of Phase 3 breakdown:
- TimeSeriesSplit with expanding window (n_splits=5)
- Purge logic using trading-day counts
- Embargo: 63 trading days (NON-NEGOTIABLE)
- Validation: ensure no overlap within embargo window

Per specs Section 1 (Sprint 0) and theory.md Section 4.3.
"""

import logging
from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NON-NEGOTIABLE constant per specs Section 1 (in trading days)
EMBARGO_DAYS = 63


@dataclass
class CPCVConfig:
    """Configuration for Purged & Embargoed Time Series Split."""

    n_splits: int = 5
    max_label_horizon: int = 63  # Maximum label horizon in trading days


class PurgedEmbargoedTimeSeriesSplit:
    """
    Expanding-window time series split with trading-day-aware purging and embargo.

    Implements:
    1. TimeSeriesSplit with expanding window (n_splits)
    2. Purge logic: remove samples from train if label window (in trading days) overlaps test
    3. Embargo: 63 trading-day gap between train and test (NON-NEGOTIABLE per specs)
    4. Validation: ensure no overlap within embargo window

    NOTE: All day counts (embargo_days, max_label_horizon) are in TRADING DAYS,
    not calendar days. This is critical for financial data with weekends/holidays.

    Attributes:
        n_splits: Number of cross-validation splits
        embargo_days: 63 trading days (hardcoded, NON-NEGOTIABLE per specs)
        max_label_horizon: Maximum forward-looking window in trading days

    Example:
        >>> cv = PurgedEmbargoedTimeSeriesSplit(n_splits=5, max_label_horizon=63)
        >>> for train_idx, test_idx in cv.split(data_df):
        ...     print(f"Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    """

    def __init__(
        self,
        n_splits: int = 5,
        max_label_horizon: int = 63,
    ):
        """
        Initialize purged & embargoed time series splitter.

        Args:
            n_splits: Number of expanding window splits
            max_label_horizon: Maximum days labels look forward (in trading days)

        Note:
            embargo_days is hardcoded to 63 trading days (NON-NEGOTIABLE per specs Section 1)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if max_label_horizon < 0:
            raise ValueError("max_label_horizon must be >= 0")

        self.n_splits = n_splits
        self.embargo_days = EMBARGO_DAYS  # NON-NEGOTIABLE, in trading days
        self.max_label_horizon = max_label_horizon

        logger.info(
            f"Initialized PurgedEmbargoedTimeSeriesSplit: n_splits={n_splits}, "
            f"embargo_days={self.embargo_days} trading days (NON-NEGOTIABLE), "
            f"max_label_horizon={max_label_horizon} trading days"
        )

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits with trading-day-aware purging and embargo.

        For n_splits folds, we divide the data into (n_splits + 1) sections:
        - Fold 0: train on section 0, test on section 1
        - Fold 1: train on sections 0-1, test on section 2
        - ...
        - Fold n-1: train on sections 0 to n-1, test on section n

        Args:
            X: DataFrame with DatetimeIndex or MultiIndex with datetime level
               Must be sorted by date
            y: Ignored (for sklearn compatibility)
            groups: Ignored (for sklearn compatibility)

        Yields:
            (train_indices, test_indices) tuples
            Indices are integer positions in the DataFrame

        Raises:
            ValueError: If X is empty, not sorted, has invalid index, or insufficient
                       data for splitting with the specified embargo
        """
        # Validate input
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        # Extract datetime index
        if isinstance(X.index, pd.MultiIndex):
            if "datetime" not in X.index.names:
                raise ValueError("MultiIndex must have 'datetime' level")
            dates = X.index.get_level_values("datetime")
        elif isinstance(X.index, pd.DatetimeIndex):
            dates = X.index
        else:
            raise ValueError("Index must be DatetimeIndex or MultiIndex with datetime level")

        # Check if sorted
        if not dates.is_monotonic_increasing:
            raise ValueError("Data must be sorted by date")

        # Get unique dates (these are trading days)
        unique_dates = pd.Series(dates.unique()).sort_values().reset_index(drop=True)
        n_dates = len(unique_dates)

        # For n_splits folds, we create n_splits + 1 sections
        # Minimum requirement: enough trading days for embargo and purging
        min_section_size = max(self.embargo_days, self.max_label_horizon) + 1
        min_total_dates_needed = min_section_size * (self.n_splits + 1)

        if n_dates < min_total_dates_needed:
            raise ValueError(
                f"Insufficient data for {self.n_splits} splits with "
                f"{self.embargo_days} trading-day embargo (NON-NEGOTIABLE). "
                f"Have {n_dates} trading days but need at least "
                f"{min_total_dates_needed} trading days "
                f"({min_section_size} trading days per section x {self.n_splits + 1} sections)"
            )

        logger.info(f"Splitting {len(X)} samples across {n_dates} unique trading days")
        logger.info(f"Date range: {unique_dates.min()} to {unique_dates.max()}")

        # Divide dates into (n_splits + 1) approximately equal sections
        section_size = n_dates / (self.n_splits + 1)
        split_points = [int(round(section_size * i)) for i in range(self.n_splits + 2)]
        split_points[0] = 0
        split_points[-1] = n_dates

        logger.debug(f"Split points (trading-day indices): {split_points}")

        for fold_idx in range(self.n_splits):
            # Fold i: train on sections 0 to i, test on section i+1
            test_start_idx = split_points[fold_idx + 1]
            test_end_idx = split_points[fold_idx + 2]

            # Get date boundaries for test period
            test_start_date = unique_dates.iloc[test_start_idx]
            test_end_date = unique_dates.iloc[test_end_idx - 1]

            # Apply purge and embargo using TRADING DAY counts
            # Purge: exclude training samples whose label window overlaps test
            purge_cutoff_idx = max(0, test_start_idx - self.max_label_horizon)

            # Embargo: additional safety buffer
            embargo_cutoff_idx = max(0, test_start_idx - self.embargo_days)

            # Combined cutoff: most restrictive (earliest index)
            train_cutoff_idx = min(purge_cutoff_idx, embargo_cutoff_idx)
            train_cutoff_date = unique_dates.iloc[train_cutoff_idx] if train_cutoff_idx < n_dates else unique_dates.iloc[0]

            # Find indices for train and test
            train_mask = (dates >= unique_dates.iloc[0]) & (dates < train_cutoff_date)
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            # Validate fold has non-empty sets
            if len(train_indices) == 0:
                raise ValueError(
                    f"Fold {fold_idx}: Empty training set after embargo/purge"
                )

            if len(test_indices) == 0:
                raise ValueError(f"Fold {fold_idx}: Empty test set")

            # Log fold info
            train_max_date = dates[train_indices].max()
            test_min_date = dates[test_indices].min()
            gap_mask = (unique_dates > train_max_date) & (unique_dates < test_min_date)
            trading_day_gap = gap_mask.sum() + 1

            logger.info(
                f"Fold {fold_idx}: train={len(train_indices)} samples, "
                f"test={len(test_indices)} samples, gap={trading_day_gap} trading days"
            )

            # Validate
            self._validate_fold(fold_idx, train_indices, test_indices, dates, unique_dates)

            yield (train_indices, test_indices)

    def _validate_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        dates: pd.Series,
        unique_dates: pd.Series,
    ) -> None:
        """Validate fold satisfies embargo and purge constraints in trading days."""
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Fold {fold_idx}: Empty train or test set")

        train_max = dates[train_idx].max()
        test_min = dates[test_idx].min()

        gap_mask = (unique_dates > train_max) & (unique_dates < test_min)
        trading_day_gap = gap_mask.sum() + 1

        if trading_day_gap <= self.embargo_days:
            raise ValueError(
                f"Fold {fold_idx}: Embargo violation - "
                f"gap={trading_day_gap} <= embargo={self.embargo_days} trading days"
            )

        if trading_day_gap <= self.max_label_horizon:
            raise ValueError(
                f"Fold {fold_idx}: Purge violation - "
                f"gap={trading_day_gap} <= horizon={self.max_label_horizon} trading days"
            )

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits (sklearn-compatible)."""
        return self.n_splits


def create_cv_from_config(config: dict) -> PurgedEmbargoedTimeSeriesSplit:
    """Create splitter from configuration dictionary."""
    cv_config = config.get("cv_scheme", {})
    labels_config = config.get("labels", {})

    n_splits = cv_config.get("n_splits", 5)

    if "embargo_days" in cv_config:
        config_embargo = cv_config["embargo_days"]
        if config_embargo != EMBARGO_DAYS:
            raise ValueError(
                f"embargo_days in config ({config_embargo}) must be {EMBARGO_DAYS} (NON-NEGOTIABLE)"
            )

    horizons = labels_config.get("horizons", [21, 63])
    max_label_horizon = max(horizons)

    return PurgedEmbargoedTimeSeriesSplit(
        n_splits=n_splits,
        max_label_horizon=max_label_horizon,
    )
