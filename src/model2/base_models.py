"""Model 2 Base Models: Trainers and CV Orchestration.

Implements BaseModelTrainer ABC, RidgeTrainer, XGBoostTrainer,
and CV training loop orchestrator.

This module is extracted from src/model2/train.py to separate
base model implementations from the CPCV splitting logic.
"""

import json
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

# ============================================================================
# Constants (P3C4-001-010)
# ============================================================================

# CV score schema keys (NON-NEGOTIABLE per P3C4-001-010)
CV_SCORE_SCHEMA_KEYS = frozenset({"model", "horizon", "fold_id", "r2", "mse", "mae"})

# Outlier threshold: ±500 bps = ±5.0 (NON-NEGOTIABLE per P3C4-001-006)
OUTLIER_THRESHOLD_BPS = 5.0


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
# CV Score Computation and Logging (P3C4-001-010)
# ============================================================================


def compute_cv_scores(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    fold_idx: int,
    model_name: str,
    horizon: str,
) -> dict[str, Any]:
    """Compute CV metrics for a single fold.

    Per P3C4-001-010: Primary function for computing fold-level CV scores.
    Validates predictions, computes r2/mse/mae, handles edge cases.

    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        fold_idx: Fold index
        model_name: Model identifier ("ridge", "xgboost")
        horizon: Horizon identifier ("21d", "63d")

    Returns:
        Dictionary with CV score schema:
        {'model': str, 'horizon': str, 'fold_id': int, 'r2': float, 'mse': float, 'mae': float}

    Raises:
        ValueError: If y_pred contains NaN or Inf values

    Edge cases:
        - y_true all constant: r2=NaN, log warning
        - y_pred contains NaN: Raise ValueError before scoring
        - Negative r2 (worse than mean): Log warning, keep value
    """
    # Normalize inputs to NumPy arrays for consistent validation
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if y_true_array.ndim != 1:
        raise ValueError(
            f"Fold {fold_idx}: y_true must be one-dimensional, got shape {y_true_array.shape}"
        )
    if y_pred_array.ndim != 1:
        raise ValueError(
            f"Fold {fold_idx}: y_pred must be one-dimensional, got shape {y_pred_array.shape}"
        )
    if y_true_array.shape[0] != y_pred_array.shape[0]:
        raise ValueError(
            f"Fold {fold_idx}: y_true and y_pred length mismatch "
            f"({y_true_array.shape[0]} vs {y_pred_array.shape[0]})"
        )

    # Validate predictions are finite
    if not np.all(np.isfinite(y_pred_array)):
        raise ValueError(f"Fold {fold_idx}: predictions contain NaN or Inf values")

    # Compute metrics
    r2 = r2_score(y_true_array, y_pred_array)
    mse = mean_squared_error(y_true_array, y_pred_array)
    mae = mean_absolute_error(y_true_array, y_pred_array)

    # Handle edge case: constant target (r2 may be NaN)
    if y_true_array.size > 0 and np.allclose(y_true_array, y_true_array[0]):
        if np.issubdtype(y_true_array.dtype, np.number):
            constant_value = f"{float(y_true_array[0]):.4f}"
        else:
            constant_value = repr(y_true_array[0])
        logger.warning(
            f"Fold {fold_idx}: constant target detected (y_true={constant_value}), r2={r2}"
        )

    # Handle edge case: negative r2 (worse than mean baseline)
    if np.isfinite(r2) and r2 < 0:
        logger.warning(
            f"Fold {fold_idx}: negative r2={r2:.4f} (model worse than mean baseline)"
        )

    # Return CV score dict with schema-compliant keys
    return {
        "model": model_name,
        "horizon": horizon,
        "fold_id": fold_idx,
        "r2": float(r2),
        "mse": float(mse),
        "mae": float(mae),
    }


def validate_cv_score_schema(cv_score: dict[str, Any]) -> None:
    """Validate CV score dict matches expected schema.

    Per P3C4-001-010: Ensure CV score dict has required keys.

    Args:
        cv_score: CV score dict from compute_cv_scores()

    Raises:
        ValueError: If required keys are missing or types are invalid
    """
    missing_keys = CV_SCORE_SCHEMA_KEYS - set(cv_score.keys())
    if missing_keys:
        raise ValueError(
            f"CV score dict missing required keys: {sorted(missing_keys)}. "
            f"Expected keys: {sorted(CV_SCORE_SCHEMA_KEYS)}"
        )

    # Type validation
    if not isinstance(cv_score["model"], str):
        raise ValueError(f"CV score 'model' must be str, got {type(cv_score['model'])}")
    if not isinstance(cv_score["horizon"], str):
        raise ValueError(f"CV score 'horizon' must be str, got {type(cv_score['horizon'])}")
    if not isinstance(cv_score["fold_id"], int):
        raise ValueError(f"CV score 'fold_id' must be int, got {type(cv_score['fold_id'])}")
    if not isinstance(cv_score["r2"], (int, float)):
        raise ValueError(f"CV score 'r2' must be numeric, got {type(cv_score['r2'])}")
    if not isinstance(cv_score["mse"], (int, float)):
        raise ValueError(f"CV score 'mse' must be numeric, got {type(cv_score['mse'])}")
    if not isinstance(cv_score["mae"], (int, float)):
        raise ValueError(f"CV score 'mae' must be numeric, got {type(cv_score['mae'])}")


def aggregate_cv_scores(cv_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate CV scores across folds.

    Per P3C4-001-010: Compute mean and std for r2, mse, mae across all folds.
    Uses np.nanmean/np.nanstd to handle NaN values (e.g., constant targets).

    Args:
        cv_scores: List of CV score dicts from compute_cv_scores()

    Returns:
        Aggregate statistics dict:
        {'n_folds': int, 'r2_mean': float, 'r2_std': float,
         'mse_mean': float, 'mse_std': float, 'mae_mean': float, 'mae_std': float}

    Raises:
        ValueError: If cv_scores is empty
    """
    if not cv_scores:
        raise ValueError("Cannot aggregate empty cv_scores list")

    # Extract metric arrays
    r2_scores = np.array([score["r2"] for score in cv_scores])
    mse_scores = np.array([score["mse"] for score in cv_scores])
    mae_scores = np.array([score["mae"] for score in cv_scores])

    # Compute statistics using np.nanmean/np.nanstd for robust handling
    return {
        "n_folds": len(cv_scores),
        "r2_mean": float(np.nanmean(r2_scores)),
        "r2_std": float(np.nanstd(r2_scores)),
        "mse_mean": float(np.nanmean(mse_scores)),
        "mse_std": float(np.nanstd(mse_scores)),
        "mae_mean": float(np.nanmean(mae_scores)),
        "mae_std": float(np.nanstd(mae_scores)),
    }


def log_cv_scores_json(cv_scores: list[dict[str, Any]], output_path: Path) -> None:
    """Write CV scores to JSON file with NaN support.

    Per P3C4-001-010: Persist CV scores to JSON with allow_nan=True for
    edge cases like constant targets (r2=NaN).

    Args:
        cv_scores: List of CV score dicts from compute_cv_scores()
        output_path: Path to output JSON file

    Raises:
        ValueError: If cv_scores is empty or schema validation fails
        OSError: If file write fails
    """
    if not cv_scores:
        raise ValueError("Cannot write empty cv_scores list to JSON")

    # Validate all scores match schema
    for score in cv_scores:
        validate_cv_score_schema(score)

    # Create parent directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to JSON with allow_nan=True for NaN handling
    try:
        with open(output_path, "w") as f:
            json.dump(cv_scores, f, indent=2, allow_nan=True)
        logger.info(f"Saved {len(cv_scores)} CV scores to {output_path}")
    except Exception as e:
        raise OSError(f"Failed to write CV scores to {output_path}: {e}") from e


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
    """Orchestrate cross-validation training loop for a single model-horizon pair.

    Per P3C4-001-006: train on each CV fold, collect OOF predictions,
    compute CV scores, train final model on all data.

    Per P3C4-001-010: Use compute_cv_scores() and log aggregate statistics.

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
        - Outlier predictions: Log warning if >1% of predictions exceed ±OUTLIER_THRESHOLD_BPS
    """
    logger.info(
        f"Starting CV training: model={model_name}, horizon={horizon}, "
        f"n_samples={len(X)}, n_features={X.shape[1]}, n_splits={cv_splitter.n_splits}"
    )

    # Initialize OOF prediction storage
    fold_predictions_list: list[tuple[Sequence[Any], Sequence[float], int]] = []

    # Initialize CV scores list
    cv_scores: list[dict[str, Any]] = []

    try:
        fold_splits = list(cv_splitter.split(X))
    except ValueError as err:
        raise ValueError(
            "CV splitter failed to generate folds that meet the minimum 100 required test "
            f"samples per fold: {err}"
        ) from err

    # Iterate through CV folds
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
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

        # Compute and log fold metrics using compute_cv_scores() (P3C4-001-010)
        fold_metrics = compute_cv_scores(y_test, predictions, fold_idx, model_name, horizon)
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

    # Log aggregate CV scores (P3C4-001-010)
    if cv_scores:
        agg_scores = aggregate_cv_scores(cv_scores)
        logger.info(
            f"CV aggregate ({agg_scores['n_folds']} folds): "
            f"r2_mean={agg_scores['r2_mean']:.4f}, r2_std={agg_scores['r2_std']:.4f}, "
            f"mse_mean={agg_scores['mse_mean']:.4f}, mse_std={agg_scores['mse_std']:.4f}, "
            f"mae_mean={agg_scores['mae_mean']:.4f}, mae_std={agg_scores['mae_std']:.4f}"
        )

    # Return results
    return {
        "oof_predictions": oof_predictions,
        "final_model": final_trainer,
        "cv_scores": cv_scores,
    }


def _clone_trainer(trainer: BaseModelTrainer) -> BaseModelTrainer:
    """Clone a trainer instance to create a fresh model for a new fold.

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
    """Compute CV metrics for a single fold.

    DEPRECATED: Use compute_cv_scores() instead (P3C4-001-010).
    Kept for backward compatibility.

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
    # Delegate to compute_cv_scores() (P3C4-001-010)
    return compute_cv_scores(y_true, y_pred, fold_idx, model_name, horizon)


def _check_outlier_predictions(predictions: np.ndarray, fold_idx: int) -> None:
    """Check for outlier predictions and log warnings.

    Per P3C4-001-006: Log warning if >1% of predictions exceed ±OUTLIER_THRESHOLD_BPS.
    Per P3C4-001-010: Use OUTLIER_THRESHOLD_BPS constant instead of hardcoded 5.0.

    Args:
        predictions: Model predictions for a fold
        fold_idx: Fold index for logging context
    """
    if len(predictions) == 0:
        return

    exceed_count = np.count_nonzero(np.abs(predictions) > OUTLIER_THRESHOLD_BPS)
    exceed_fraction = exceed_count / len(predictions)
    if exceed_fraction > 0.01:
        if math.isclose(exceed_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
            logger.warning(
                f"Fold {fold_idx}: all predictions exceed ±{OUTLIER_THRESHOLD_BPS:.1f} "
                f"(count={exceed_count}, max={predictions.max():.2f}, min={predictions.min():.2f})"
            )
        else:
            logger.warning(
                f"Fold {fold_idx}: {exceed_fraction:.2%} of predictions exceed ±{OUTLIER_THRESHOLD_BPS:.1f} "
                f"(count={exceed_count}, max={predictions.max():.2f}, min={predictions.min():.2f})"
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
        # Preserve MultiIndex structure if present
        if isinstance(indices, pd.MultiIndex):
            index = indices
        else:
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


# ============================================================================
# Multi-Horizon Training Wrapper (P3C4-001-009)
# ============================================================================


def train_multi_horizon(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cv_splitter: Any,
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Train all base models for both 21d and 63d horizons.

    Per P3C4-001-009: Orchestrate training across all model-horizon pairs,
    handling edge cases (missing labels, index mismatches, partial failures).

    Args:
        features: Feature matrix with MultiIndex (instrument, datetime)
        labels: Labels DataFrame with columns [label_21d, label_63d]
        cv_splitter: CPCV splitter instance
        config: Configuration dict with keys [models, horizons, ...]

    Returns:
        Dict mapping (model_name, horizon) -> {
            'oof': DataFrame,
            'model': BaseModelTrainer,
            'cv_scores': List[dict]
        }

    Raises:
        KeyError: If required label columns are missing
        ValueError: If feature-label index mismatch or all models fail
        RuntimeError: If all model-horizon pairs fail training

    Edge cases:
        - Label column missing: Raise KeyError with expected columns
        - Feature-label index mismatch: Inner join, log dropped count
        - Horizon fails mid-training: Log error, continue with other horizons
        - All models fail: Raise RuntimeError with summary

    Example:
        >>> results = train_multi_horizon(features, labels, cv_splitter, config)
        >>> ridge_21d_oof = results[('ridge', '21d')]['oof']
        >>> xgb_63d_model = results[('xgboost', '63d')]['model']
    """
    logger.info("Starting multi-horizon training orchestration")

    # Validate label columns exist
    required_label_cols = {"label_21d", "label_63d"}
    missing_labels = required_label_cols - set(labels.columns)
    if missing_labels:
        raise KeyError(
            f"Missing required label columns: {sorted(missing_labels)}. "
            f"Expected columns: {sorted(required_label_cols)}"
        )

    # Validate features and labels have compatible indices via inner join
    original_feature_count = len(features)
    original_label_count = len(labels)

    # Perform inner join to align features and labels
    aligned_features = features.join(labels[[]], how="inner")
    aligned_labels = labels.loc[aligned_features.index]

    dropped_rows = original_feature_count - len(aligned_features)
    if dropped_rows > 0:
        logger.warning(
            f"Dropped {dropped_rows} rows due to feature-label index mismatch "
            f"(features: {original_feature_count}, labels: {original_label_count}, "
            f"aligned: {len(aligned_features)})"
        )

    # Extract model names from config (default to ridge and xgboost)
    model_names = config.get("models", ["ridge", "xgboost"])
    if not isinstance(model_names, list) or not model_names:
        model_names = ["ridge", "xgboost"]
        logger.warning(f"Config 'models' key missing or invalid, using defaults: {model_names}")

    # Extract horizons from config or default to 21d and 63d
    horizons = config.get("horizons", ["21d", "63d"])
    if not isinstance(horizons, list) or not horizons:
        horizons = ["21d", "63d"]
        logger.warning(f"Config 'horizons' key missing or invalid, using defaults: {horizons}")

    # Model registry: map string names to trainer classes
    model_registry = {
        "ridge": RidgeTrainer,
        "xgboost": XGBoostTrainer,
    }

    # Results storage and failure tracking
    results: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[tuple[str, str, str]] = []  # (model_name, horizon, error_msg)

    # Iterate over all (model_name, horizon) pairs
    total_pairs = len(model_names) * len(horizons)
    logger.info(
        f"Training {total_pairs} model-horizon pairs: models={model_names}, horizons={horizons}"
    )

    for model_name in model_names:
        for horizon in horizons:
            logger.info(f"Training pair: model={model_name}, horizon={horizon}")

            try:
                # Get trainer class from registry
                if model_name not in model_registry:
                    raise ValueError(
                        f"Unknown model name '{model_name}'. "
                        f"Available models: {list(model_registry.keys())}"
                    )

                trainer_class = model_registry[model_name]
                trainer = trainer_class()

                # Extract label series for this horizon
                label_col = f"label_{horizon}"
                if label_col not in aligned_labels.columns:
                    raise KeyError(
                        f"Label column '{label_col}' not found in labels DataFrame. "
                        f"Available columns: {list(aligned_labels.columns)}"
                    )

                y = aligned_labels[label_col]

                # Run CV training for this model-horizon pair
                cv_result = run_cv_training(
                    X=aligned_features,
                    y=y,
                    cv_splitter=cv_splitter,
                    trainer=trainer,
                    model_name=model_name,
                    horizon=horizon,
                )

                # Store successful result
                results[(model_name, horizon)] = {
                    "oof": cv_result["oof_predictions"],
                    "model": cv_result["final_model"],
                    "cv_scores": cv_result["cv_scores"],
                }

                logger.info(
                    f"Successfully trained {model_name} for {horizon} horizon "
                    f"(oof_size={len(cv_result['oof_predictions'])}, "
                    f"cv_folds={len(cv_result['cv_scores'])})"
                )

            except Exception as e:
                error_msg = str(e)
                failures.append((model_name, horizon, error_msg))
                logger.error(
                    f"Training failed for model={model_name}, horizon={horizon}: {error_msg}"
                )
                # Continue with other pairs (partial failure handling)

    # Validate at least one pair succeeded
    n_success = len(results)
    n_failed = len(failures)

    if n_success == 0:
        # All pairs failed - raise RuntimeError with summary
        failure_summary = "\n".join(
            [f"  - {model}/{horizon}: {err}" for model, horizon, err in failures]
        )
        raise RuntimeError(
            f"All {total_pairs} model-horizon pairs failed training:\n{failure_summary}"
        )

    # Log summary
    logger.info(
        f"Multi-horizon training complete: {n_success}/{total_pairs} pairs successful, "
        f"{n_failed} failed"
    )

    if failures:
        logger.warning(f"Failed pairs: {[(m, h) for m, h, _ in failures]}")

    return results


def save_multi_horizon_results(
    results: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    region: str,
) -> None:
    """Save all multi-horizon training outputs to disk.

    Per P3C4-001-009: Persist OOF predictions, models, CV scores, and feature
    importance for all model-horizon pairs.

    Per P3C4-001-010: Use log_cv_scores_json() for JSON output with NaN support.

    Args:
        results: Output from train_multi_horizon()
        output_dir: Base directory for outputs (e.g., data/model2/us/)
        region: Region identifier ("US" or "CN")

    Raises:
        OSError: If any file write operations fail
        ValueError: If results dict is empty

    Edge cases:
        - Empty results: Raise ValueError
        - Output directory doesn't exist: Create with parents
        - Partial save failure: Log error, continue with remaining outputs

    Output structure:
        {output_dir}/
            oof/
                ridge_21d_oof.parquet
                ridge_63d_oof.parquet
                xgboost_21d_oof.parquet
                xgboost_63d_oof.parquet
            models/
                ridge_21d.pkl
                ridge_63d.pkl
                xgboost_21d.pkl
                xgboost_63d.pkl
            cv_scores/
                ridge_21d_cv_scores.json
                ridge_63d_cv_scores.json
                xgboost_21d_cv_scores.json
                xgboost_63d_cv_scores.json
            feature_importance/
                xgboost_21d_importance.parquet
                xgboost_63d_importance.parquet
    """
    # Import persistence utilities here to avoid circular imports
    from src.model2.persistence import save_oof_predictions, save_trained_model

    # Validate results dict is not empty
    if not results:
        raise ValueError("Cannot save empty results dictionary")

    logger.info(
        f"Saving multi-horizon results for {len(results)} model-horizon pairs to {output_dir}"
    )

    # Create output subdirectories
    output_dir = Path(output_dir)
    subdirs = {
        "oof": output_dir / "oof",
        "models": output_dir / "models",
        "cv_scores": output_dir / "cv_scores",
        "feature_importance": output_dir / "feature_importance",
    }

    for subdir_name, subdir_path in subdirs.items():
        subdir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {subdir_path}")

    # Track save successes and failures
    save_failures: list[tuple[str, str, str, str]] = []  # (model, horizon, artifact, error)

    # Iterate over all (model_name, horizon) pairs
    for (model_name, horizon), pair_results in results.items():
        logger.info(f"Saving outputs for {model_name}_{horizon}")

        # 1. Save OOF predictions
        try:
            oof_path = subdirs["oof"] / f"{model_name}_{horizon}_oof.parquet"
            save_oof_predictions(pair_results["oof"], oof_path)
            logger.debug(f"Saved OOF predictions to {oof_path}")
        except Exception as e:
            error_msg = f"Failed to save OOF predictions: {e}"
            save_failures.append((model_name, horizon, "oof", error_msg))
            logger.error(f"{model_name}_{horizon}: {error_msg}")

        # 2. Save final model
        try:
            model_path = subdirs["models"] / f"{model_name}_{horizon}.pkl"
            save_trained_model(
                pair_results["model"],
                model_path,
                extra_metadata={"region": region, "horizon": horizon},
            )
            logger.debug(f"Saved model to {model_path}")
        except Exception as e:
            error_msg = f"Failed to save model: {e}"
            save_failures.append((model_name, horizon, "model", error_msg))
            logger.error(f"{model_name}_{horizon}: {error_msg}")

        # 3. Save CV scores to JSON (P3C4-001-010: use log_cv_scores_json)
        try:
            cv_scores_path = subdirs["cv_scores"] / f"{model_name}_{horizon}_cv_scores.json"
            log_cv_scores_json(pair_results["cv_scores"], cv_scores_path)
            logger.debug(f"Saved CV scores to {cv_scores_path}")
        except Exception as e:
            error_msg = f"Failed to save CV scores: {e}"
            save_failures.append((model_name, horizon, "cv_scores", error_msg))
            logger.error(f"{model_name}_{horizon}: {error_msg}")

        # 4. Save feature importance (XGBoost only)
        if model_name == "xgboost":
            try:
                model = pair_results["model"]
                importance_df = model.get_feature_importance()

                if importance_df is not None and not importance_df.empty:
                    importance_path = (
                        subdirs["feature_importance"] / f"{model_name}_{horizon}_importance.parquet"
                    )
                    save_feature_importance(importance_df, importance_path)
                    logger.debug(f"Saved feature importance to {importance_path}")
                else:
                    logger.warning(
                        f"{model_name}_{horizon}: No feature importance to save (empty or None)"
                    )
            except Exception as e:
                error_msg = f"Failed to save feature importance: {e}"
                save_failures.append((model_name, horizon, "feature_importance", error_msg))
                logger.error(f"{model_name}_{horizon}: {error_msg}")

    # Log summary
    total_artifacts = len(results) * 3 + sum(
        1 for (model_name, _) in results.keys() if model_name == "xgboost"
    )
    n_failures = len(save_failures)
    n_success = total_artifacts - n_failures

    logger.info(
        f"Saved outputs for {len(results)} model-horizon pairs to {output_dir}: "
        f"{n_success}/{total_artifacts} artifacts saved successfully"
    )

    if save_failures:
        logger.warning(
            f"{n_failures} artifact save failures: {[(m, h, a) for m, h, a, _ in save_failures]}"
        )
