"""Model 2 Base Models: Trainers and CV Orchestration.

Implements BaseModelTrainer ABC, RidgeTrainer, XGBoostTrainer,
and CV training loop orchestrator.

This module is extracted from src/model2/train.py to separate
base model implementations from the CPCV splitting logic.
"""

import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


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

        Per P3C4-001-007: extract gain and weight importance, log warnings for
        edge cases (no features, zero importance), and log top 20 features at INFO level.

        Returns:
            DataFrame with columns [feature, importance_gain, importance_weight]
            Sorted by importance_gain descending

        Raises:
            RuntimeError: If model not fitted
        """
        if not self._is_fitted:
            raise RuntimeError(
                "XGBoostTrainer must be fitted before extracting feature importance."
            )

        booster = self.model.get_booster()

        gain_scores = booster.get_score(importance_type="gain")
        weight_scores = booster.get_score(importance_type="weight")

        # Edge case: No features used (constant target or model did not use any splits)
        if not gain_scores and not weight_scores:
            logger.warning(
                "No features used by XGBoost model. This may indicate a constant target "
                "or insufficient training data."
            )
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

        # Edge case: All features have zero importance
        if (df["importance_gain"] == 0.0).all():
            logger.warning(
                "All features have zero importance_gain. This may indicate model underfitting "
                "or improper feature encoding."
            )

        # Log top 20 features at INFO level
        top_n = min(20, len(df))
        if top_n > 0:
            logger.info(f"Top {top_n} features by importance_gain:")
            for idx in range(top_n):
                row = df.iloc[idx]
                logger.info(
                    f"  {idx + 1}. {row['feature']}: gain={row['importance_gain']:.4f}, "
                    f"weight={row['importance_weight']:.0f}"
                )

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


def save_feature_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    """Save feature importance DataFrame to parquet file.

    Per P3C4-001-007: Helper function to persist feature importance to disk.

    Args:
        importance_df: DataFrame with columns [feature, importance_gain, importance_weight]
        output_path: Path to output parquet file

    Raises:
        ValueError: If DataFrame schema is invalid
        OSError: If file write fails

    Edge cases:
        - Output directory doesn't exist: Create parent directories
        - File already exists: Overwrite with warning
        - Empty DataFrame: Write empty parquet (valid edge case)
    """
    # Validate schema
    required_columns = {"feature", "importance_gain", "importance_weight"}
    if not required_columns.issubset(importance_df.columns):
        missing = required_columns - set(importance_df.columns)
        raise ValueError(
            f"Invalid feature importance schema. Missing columns: {missing}. "
            f"Expected columns: {required_columns}"
        )

    # Create parent directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Warn if overwriting existing file
    if output_path.exists():
        logger.warning(f"Overwriting existing feature importance file: {output_path}")

    # Write to parquet
    try:
        importance_df.to_parquet(output_path, index=False)
        logger.info(f"Saved feature importance ({len(importance_df)} features) to {output_path}")
    except Exception as e:
        raise OSError(f"Failed to write feature importance to {output_path}: {e}") from e


# ============================================================================
# CV Training Loop Orchestrator (P3C4-001-006)
# ============================================================================


def run_cv_training(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splitter: Any,  # PurgedEmbargoedTimeSeriesSplit
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
        - Outlier predictions: Log warning if all predictions exceed ±1000 bps
    """
    logger.info(
        f"Starting CV training: model={model_name}, horizon={horizon}, "
        f"n_samples={len(X)}, n_features={X.shape[1]}, n_splits={cv_splitter.n_splits}"
    )

    # Initialize OOF prediction storage
    fold_predictions_list: list[tuple[Sequence[Any], Sequence[float], int]] = []

    # Initialize CV scores list
    cv_scores: list[dict[str, Any]] = []

    # Iterate through CV folds
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        logger.info(f"Fold {fold_idx}: train_size={len(train_idx)}, test_size={len(test_idx)}")

        # Validate fold size >= 100 samples
        if len(test_idx) < 100:
            raise ValueError(
                f"Fold {fold_idx} has {len(test_idx)} test samples, minimum 100 required"
            )

        # Clone trainer for this fold (fresh model instance)
        fold_trainer = _clone_trainer(trainer)

        # Extract train/test data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Fit on train data
        try:
            fold_trainer.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Fold {fold_idx} training failed: {e}")
            raise ValueError(f"Fold {fold_idx} training failed") from e

        # Predict on test data
        predictions = fold_trainer.predict(X_test)

        # CRITICAL FIX: Store predictions with X_test.index to preserve MultiIndex
        # This ensures downstream joins work correctly with (instrument, datetime) labels
        fold_predictions_list.append((X_test.index, predictions, fold_idx))

        # Compute and log fold metrics
        fold_metrics = _compute_fold_metrics(
            y_test, predictions, fold_idx, model_name, horizon
        )
        cv_scores.append(fold_metrics)
        logger.info(
            f"Fold {fold_idx} metrics: r2={fold_metrics['r2']:.4f}, "
            f"mse={fold_metrics['mse']:.4f}, mae={fold_metrics['mae']:.4f}"
        )

        # Check for outlier predictions
        _check_outlier_predictions(predictions, fold_idx)

    # Aggregate OOF predictions
    oof_predictions = aggregate_oof_predictions(fold_predictions_list)
    logger.info(f"Aggregated {len(oof_predictions)} OOF predictions across all folds")

    # Train final model on all data
    logger.info(f"Training final model on all {len(X)} samples")
    final_trainer = _clone_trainer(trainer)
    try:
        final_trainer.fit(X, y)
    except Exception as e:
        logger.error(f"Final model training failed: {e}")
        raise ValueError("Final model training failed") from e

    # Log aggregate CV scores
    if cv_scores:
        r2_scores = [score["r2"] for score in cv_scores]
        logger.info(
            f"CV aggregate: r2_mean={np.mean(r2_scores):.4f}, r2_std={np.std(r2_scores):.4f}"
        )

    # Return results
    return {
        "oof_predictions": oof_predictions,
        "final_model": final_trainer,
        "cv_scores": cv_scores,
    }


def _clone_trainer(trainer: BaseModelTrainer) -> BaseModelTrainer:
    """
    Clone a trainer instance to create a fresh model for a new fold.

    Per P3C4-001-006: Each fold should use a fresh model instance.
    Uses trainer.__class__() to preserve actual class (including subclasses).

    Args:
        trainer: Original trainer instance

    Returns:
        New trainer instance of the same class with default hyperparameters

    Note:
        This preserves subclass semantics, critical for testing with mock
        trainers like FailingTrainer(RidgeTrainer).
    """
    return trainer.__class__()


def _compute_fold_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    fold_idx: int,
    model_name: str,
    horizon: str,
) -> dict[str, Any]:
    """
    Compute CV metrics for a single fold.

    Per P3C4-001-006: Compute r2, mse, mae for each fold.

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        fold_idx: Fold index
        model_name: Model identifier
        horizon: Horizon identifier

    Returns:
        Dictionary with fold metrics: {model, horizon, fold_id, r2, mse, mae}

    Raises:
        ValueError: If y_pred contains NaN values
    """
    # Validate predictions are finite
    if not np.all(np.isfinite(y_pred)):
        raise ValueError(f"Fold {fold_idx}: predictions contain NaN or Inf values")

    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Handle edge case: constant target (r2 may be NaN)
    if np.allclose(y_true, y_true.iloc[0]):
        logger.warning(f"Fold {fold_idx}: constant target detected, r2 may be NaN")

    # Return metrics dict
    return {
        "model": model_name,
        "horizon": horizon,
        "fold_id": fold_idx,
        "r2": float(r2),
        "mse": float(mse),
        "mae": float(mae),
    }


def _check_outlier_predictions(predictions: np.ndarray, fold_idx: int) -> None:
    """
    Check for outlier predictions and log warnings.

    Per P3C4-001-006: Log warning if all predictions exceed ±1000 bps (±10.0).

    Args:
        predictions: Model predictions for a fold
        fold_idx: Fold index for logging context
    """
    if len(predictions) == 0:
        return

    outlier_threshold = 10.0
    if np.all(np.abs(predictions) > outlier_threshold):
        logger.warning(
            f"Fold {fold_idx}: all predictions exceed ±{outlier_threshold} "
            f"(max={predictions.max():.2f}, min={predictions.min():.2f})"
        )


def aggregate_oof_predictions(
    fold_predictions: Iterable[tuple[Sequence[Any], Sequence[float], int]],
) -> pd.DataFrame:
    """Aggregate out-of-fold predictions from individual cross-validation folds.

    Args:
        fold_predictions: Iterable of tuples ``(indices, predictions, fold_id)`` where
            ``indices`` is a 1-D sequence of sample indices (can be MultiIndex),
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
