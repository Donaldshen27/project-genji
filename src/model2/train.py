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
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
        self, X: pd.DataFrame, y=None, groups=None
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
            train_cutoff_date = (
                unique_dates.iloc[train_cutoff_idx]
                if train_cutoff_idx < n_dates
                else unique_dates.iloc[0]
            )

            # Find indices for train and test
            train_mask = (dates >= unique_dates.iloc[0]) & (dates < train_cutoff_date)
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            # Validate fold has non-empty sets
            if len(train_indices) == 0:
                raise ValueError(f"Fold {fold_idx}: Empty training set after embargo/purge")

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


# ============================================================================
# Chunk 4: Base Model Training Stubs
# ============================================================================
# NOTE: sklearn.linear_model.Ridge and xgboost will be imported during implementation


class BaseModelTrainer(ABC):
    """
    Abstract base class for model trainers.

    Provides consistent interface for Ridge, XGBoost, and future models.
    Per P3C4-001-001: define abstract methods for training and prediction.

    Methods:
        fit(X, y): Train the model on features X and labels y
        predict(X): Generate predictions for features X
        get_feature_importance(): Return feature importance (or None)
        get_params(): Return model hyperparameters
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModelTrainer":
        """Train the model.

        Args:
            X: Feature matrix (N_samples, N_features)
            y: Target labels (N_samples,)

        Returns:
            Self for chaining

        Raises:
            ValueError: If X or y are empty or contain invalid values
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix (N_samples, N_features)

        Returns:
            Array of predictions (N_samples,)

        Raises:
            RuntimeError: If model not fitted
        """
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame | None:
        """Extract feature importance.

        Returns:
            DataFrame with columns [feature, importance_gain, importance_weight]
            Returns None if model does not support feature importance
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters.

        Returns:
            Dictionary of hyperparameters
        """
        raise NotImplementedError


class RidgeTrainer(BaseModelTrainer):
    """
    Ridge regression trainer with frozen hyperparameters.

    Per P3C4-001-002: alpha=3.0, random_state=42 (NON-NEGOTIABLE)

    """

    ALPHA: float = 3.0
    RANDOM_STATE: int = 42

    def __init__(self, alpha: float = ALPHA, random_state: int = RANDOM_STATE):
        """Initialize Ridge trainer with frozen hyperparameters."""
        if not math.isclose(alpha, self.ALPHA, abs_tol=1e-12):
            raise ValueError(
                f"RidgeTrainer hyperparameter alpha is frozen at {self.ALPHA} (got {alpha})."
            )
        if random_state != self.RANDOM_STATE:
            raise ValueError(
                f"RidgeTrainer hyperparameter random_state is frozen at {self.RANDOM_STATE} (got {random_state})."
            )

        self.model = Ridge(alpha=self.ALPHA, random_state=self.RANDOM_STATE)
        self._is_fitted = False

        logger.debug(
            "Initialized RidgeTrainer with alpha=%s, random_state=%s (frozen values).",
            self.ALPHA,
            self.RANDOM_STATE,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeTrainer":
        """Train Ridge model.

        Edge cases:
        - Empty training set: Raise ValueError
        - NaN in X or y: sklearn raises, propagate
        - Singular matrix: Ridge regularization prevents this
        """
        if X.empty:
            raise ValueError("Cannot fit Ridge model on empty training set.")
        if y.empty:
            raise ValueError("Cannot fit Ridge model on empty target values.")
        if len(X) != len(y):
            raise ValueError(
                f"Feature matrix and target vector must have matching lengths (X={len(X)}, y={len(y)})."
            )

        self.model.fit(X, y)
        self._is_fitted = True

        logger.debug(
            "Fitted RidgeTrainer on %s samples and %s features.",
            X.shape[0],
            X.shape[1],
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate Ridge predictions."""
        if not self._is_fitted:
            raise RuntimeError("RidgeTrainer must be fitted before calling predict().")

        predictions = self.model.predict(X)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame | None:
        """Ridge does not have feature importance.

        Returns:
            None (Ridge coefficients exist but not standardized as 'importance')
        """
        return None

    def get_params(self) -> dict[str, Any]:
        """Return Ridge hyperparameters."""
        return {
            "alpha": self.ALPHA,
            "random_state": self.RANDOM_STATE,
        }


class XGBoostTrainer(BaseModelTrainer):
    """
    XGBoost regression trainer with frozen hyperparameters.

    Per P3C4-001-003 and specs:
    - max_depth=6
    - n_estimators=400
    - learning_rate=0.05 (eta)
    - subsample=0.8
    - colsample_bytree=0.8
    - random_state=42
    All parameters NON-NEGOTIABLE per specs Section 1.

    Uses xgboost.XGBRegressor with tree_method='hist' for memory efficiency.
    Sanitizes feature names by replacing special characters with underscores.
    """

    MAX_DEPTH: int = 6
    N_ESTIMATORS: int = 400
    LEARNING_RATE: float = 0.05
    SUBSAMPLE: float = 0.8
    COLSAMPLE_BYTREE: float = 0.8
    RANDOM_STATE: int = 42

    def __init__(
        self,
        max_depth: int = 6,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """Initialize XGBoost trainer with frozen hyperparameters.

        Args:
            max_depth: Maximum tree depth (default 6)
            n_estimators: Number of boosting rounds (default 400)
            learning_rate: Step size shrinkage (default 0.05)
            subsample: Subsample ratio of training instances (default 0.8)
            colsample_bytree: Subsample ratio of features (default 0.8)
            random_state: Random seed (default 42)

        Raises:
            ValueError: If any hyperparameter does not match frozen value
        """
        import xgboost as xgb

        # Validate frozen hyperparameters
        if max_depth != self.MAX_DEPTH:
            raise ValueError(
                f"XGBoostTrainer hyperparameter max_depth is frozen at {self.MAX_DEPTH} (got {max_depth})."
            )
        if n_estimators != self.N_ESTIMATORS:
            raise ValueError(
                f"XGBoostTrainer hyperparameter n_estimators is frozen at {self.N_ESTIMATORS} (got {n_estimators})."
            )
        if not math.isclose(learning_rate, self.LEARNING_RATE, abs_tol=1e-12):
            raise ValueError(
                f"XGBoostTrainer hyperparameter learning_rate is frozen at {self.LEARNING_RATE} (got {learning_rate})."
            )
        if not math.isclose(subsample, self.SUBSAMPLE, abs_tol=1e-12):
            raise ValueError(
                f"XGBoostTrainer hyperparameter subsample is frozen at {self.SUBSAMPLE} (got {subsample})."
            )
        if not math.isclose(colsample_bytree, self.COLSAMPLE_BYTREE, abs_tol=1e-12):
            raise ValueError(
                f"XGBoostTrainer hyperparameter colsample_bytree is frozen at {self.COLSAMPLE_BYTREE} (got {colsample_bytree})."
            )
        if random_state != self.RANDOM_STATE:
            raise ValueError(
                f"XGBoostTrainer hyperparameter random_state is frozen at {self.RANDOM_STATE} (got {random_state})."
            )

        self.model = xgb.XGBRegressor(
            max_depth=self.MAX_DEPTH,
            n_estimators=self.N_ESTIMATORS,
            learning_rate=self.LEARNING_RATE,
            subsample=self.SUBSAMPLE,
            colsample_bytree=self.COLSAMPLE_BYTREE,
            random_state=self.RANDOM_STATE,
            tree_method="hist",
        )
        self._is_fitted = False
        self._feature_name_mapping: dict[str, str] = {}
        self._sanitized_feature_order: list[str] = []

        logger.debug(
            "Initialized XGBoostTrainer with max_depth=%s, n_estimators=%s, learning_rate=%s, "
            "subsample=%s, colsample_bytree=%s, random_state=%s (frozen values).",
            self.MAX_DEPTH,
            self.N_ESTIMATORS,
            self.LEARNING_RATE,
            self.SUBSAMPLE,
            self.COLSAMPLE_BYTREE,
            self.RANDOM_STATE,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostTrainer":
        """Train XGBoost model.

        Edge cases:
        - Empty training set: Raise ValueError
        - NaN in y: XGBoost raises, propagate
        - Feature names with special chars: Sanitize before training
        - Duplicate sanitized names: Raise ValueError with collision details
        """
        if X.empty:
            raise ValueError("Cannot fit XGBoost model on empty training set.")
        if y.empty:
            raise ValueError("Cannot fit XGBoost model on empty target values.")
        if len(X) != len(y):
            raise ValueError(
                f"Feature matrix and target vector must have matching lengths (X={len(X)}, y={len(y)})."
            )

        sanitized_columns = []
        sanitized_to_original: dict[str, list[str]] = {}

        for col in X.columns:
            original = str(col)
            sanitized = re.sub(r"[^A-Za-z0-9_]", "_", original)
            sanitized_columns.append(sanitized)

            if sanitized not in sanitized_to_original:
                sanitized_to_original[sanitized] = []
            sanitized_to_original[sanitized].append(original)

        duplicates = {
            san: orig_list for san, orig_list in sanitized_to_original.items() if len(orig_list) > 1
        }
        if duplicates:
            collision_info = []
            for san, orig_list in list(duplicates.items())[:3]:
                collision_info.append(f"{orig_list} -> '{san}'")
            raise ValueError(
                f"Feature name sanitization produced {len(duplicates)} duplicate(s). "
                f"XGBoost requires unique feature names. Collisions: {'; '.join(collision_info)}"
            )

        self._feature_name_mapping = {
            san: orig_list[0] for san, orig_list in sanitized_to_original.items()
        }
        self._sanitized_feature_order = sanitized_columns.copy()

        X_sanitized = X.copy()
        X_sanitized.columns = sanitized_columns

        self.model.fit(X_sanitized, y)
        self._is_fitted = True

        logger.debug(
            "Fitted XGBoostTrainer on %s samples and %s features.",
            X.shape[0],
            X.shape[1],
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate XGBoost predictions."""
        if not self._is_fitted:
            raise RuntimeError("XGBoostTrainer must be fitted before calling predict().")

        sanitized_columns = []
        for col in X.columns:
            sanitized_columns.append(re.sub(r"[^A-Za-z0-9_]", "_", str(col)))

        X_sanitized = X.copy()
        X_sanitized.columns = sanitized_columns

        if self._sanitized_feature_order:
            missing_features = set(self._sanitized_feature_order) - set(X_sanitized.columns)
            unexpected_features = set(X_sanitized.columns) - set(self._sanitized_feature_order)
            if missing_features or unexpected_features:
                raise ValueError(
                    "Prediction features must match training features exactly. "
                    f"Missing: {sorted(missing_features)}; Unexpected: {sorted(unexpected_features)}"
                )
            X_sanitized = X_sanitized[self._sanitized_feature_order]

        predictions = self.model.predict(X_sanitized)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame | None:
        """Extract XGBoost feature importance.

        Per P3C4-001-007: extract gain and weight importance.

        Returns:
            DataFrame with columns [feature, importance_gain, importance_weight]
            Sorted by importance_gain descending
        """
        if not self._is_fitted:
            raise RuntimeError(
                "XGBoostTrainer must be fitted before extracting feature importance."
            )

        booster = self.model.get_booster()

        gain_scores = booster.get_score(importance_type="gain")
        weight_scores = booster.get_score(importance_type="weight")

        if not gain_scores and not weight_scores:
            return pd.DataFrame(columns=["feature", "importance_gain", "importance_weight"])

        all_features = set(gain_scores.keys()) | set(weight_scores.keys())
        rows = []
        for sanitized_feature in all_features:
            original_name = self._feature_name_mapping.get(sanitized_feature, sanitized_feature)
            rows.append(
                {
                    "feature": original_name,
                    "importance_gain": gain_scores.get(sanitized_feature, 0.0),
                    "importance_weight": weight_scores.get(sanitized_feature, 0.0),
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)

        return df

    def get_params(self) -> dict[str, Any]:
        """Return XGBoost hyperparameters."""
        return {
            "max_depth": self.MAX_DEPTH,
            "n_estimators": self.N_ESTIMATORS,
            "learning_rate": self.LEARNING_RATE,
            "subsample": self.SUBSAMPLE,
            "colsample_bytree": self.COLSAMPLE_BYTREE,
            "random_state": self.RANDOM_STATE,
        }


def run_cv_training(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splitter: PurgedEmbargoedTimeSeriesSplit,
    trainer: BaseModelTrainer,
    model_name: str,
    horizon: str,
) -> dict[str, Any]:
    """
    Orchestrate cross-validation training loop for a single model-horizon pair.

    Per P3C4-001-006: train on each CV fold, collect OOF predictions,
    compute CV scores, train final model on all data.

    Args:
        X: Feature matrix with MultiIndex (instrument, datetime)
        y: Target labels with MultiIndex (instrument, datetime)
        cv_splitter: CPCV splitter from Chunk 3
        trainer: BaseModelTrainer instance (RidgeTrainer or XGBoostTrainer)
        model_name: Model identifier ("ridge" or "xgboost")
        horizon: Horizon identifier ("21d" or "63d")

    Returns:
        Dictionary with keys:
        - 'oof_predictions': DataFrame with columns [prediction, fold_id]
        - 'final_model': Trained BaseModelTrainer on full data
        - 'cv_scores': List of dicts with per-fold metrics

    Raises:
        ValueError: If CV fold has < 100 samples or training fails

    Edge cases:
        - Small fold: Validate min 100 samples per fold
        - Training failure: Log error with fold context, re-raise
        - Outlier predictions: Log warning if >1% exceed ±500 bps
    """
    # TODO: Initialize OOF prediction storage
    # TODO: Initialize CV scores list
    # TODO: For each fold from cv_splitter.split(X):
    #   - Extract train/test indices
    #   - Validate fold size >= 100
    #   - Clone trainer for fold
    #   - Fit on train data
    #   - Predict on test data
    #   - Store predictions with fold_id
    #   - Compute and log fold metrics (r2, mse, mae)
    # TODO: Aggregate OOF predictions (call aggregate_oof_predictions)
    # TODO: Train final model on all data
    # TODO: Return results dict
    raise NotImplementedError("P3C4-001-006: run_cv_training implementation pending")


def aggregate_oof_predictions(
    fold_predictions: Iterable[tuple[Sequence[Any], Sequence[float], int]],
) -> pd.DataFrame:
    """Aggregate out-of-fold predictions from individual cross-validation folds.

    Args:
        fold_predictions: Iterable of tuples ``(indices, predictions, fold_id)`` where
            ``indices`` is a 1-D sequence of sample indices,
            ``predictions`` is a 1-D sequence aligned with ``indices``,
            and ``fold_id`` identifies the fold.

    Returns:
        DataFrame indexed by the provided ``indices`` with columns ``prediction`` and
        ``fold_id``.

    Raises:
        ValueError: If no folds are provided, shapes mismatch, indices overlap, or
            predictions contain NaN/Inf values.
    """
    entries = list(fold_predictions)
    if not entries:
        raise ValueError("fold_predictions cannot be empty.")

    frames: list[pd.DataFrame] = []
    seen_indices: set[Any] = set()

    for entry in entries:
        if len(entry) != 3:
            raise ValueError(
                "Each fold prediction entry must be a tuple of (indices, predictions, fold_id)."
            )

        indices, predictions, fold_id = entry
        index = pd.Index(indices)
        preds_array = np.asarray(predictions, dtype=np.float32)

        if preds_array.ndim != 1:
            raise ValueError(f"Predictions for fold {fold_id} must be one-dimensional.")
        if len(index) != len(preds_array):
            raise ValueError(
                f"Fold {fold_id} has mismatched indices ({len(index)}) and predictions "
                f"({len(preds_array)})."
            )
        if not np.isfinite(preds_array).all():
            raise ValueError(f"Fold {fold_id} contains NaN or infinite predictions.")

        index_list = index.tolist()
        duplicate_indices = set(index_list) & seen_indices
        if duplicate_indices:
            duplicates_preview = list(duplicate_indices)[:3]
            raise ValueError(
                f"Detected overlapping OOF indices between folds: {duplicates_preview}"
            )

        seen_indices.update(index_list)

        fold_frame = pd.DataFrame({"prediction": preds_array}, index=index)
        fold_frame["fold_id"] = np.int8(fold_id)
        frames.append(fold_frame)

        logger.debug(
            "Aggregated %s predictions for fold %s.",
            len(fold_frame),
            fold_id,
        )

    aggregated = pd.concat(frames).sort_index()
    aggregated = aggregated[["prediction", "fold_id"]]
    return aggregated
