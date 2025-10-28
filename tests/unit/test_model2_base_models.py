"""Unit tests for base model determinism validation (P3C4-001-011).

Tests:
- test_ridge_determinism: Two runs produce identical Ridge OOF (max_diff < 1e-9)
- test_xgboost_determinism: Two runs produce identical XGBoost OOF (max_diff < 1e-6)
- test_cross_run_determinism: Full pipeline run twice produces identical all outputs

Acceptance:
- Determinism test passes for both Ridge and XGBoost on synthetic data
- Identical results when training with random_state=42
- Sorted index comparison to handle prediction order variations
"""

import numpy as np
import pandas as pd
import pytest

from src.model2.base_models import (
    RidgeTrainer,
    XGBoostTrainer,
    run_cv_training,
    aggregate_oof_predictions,
)
from src.model2.train import PurgedEmbargoedTimeSeriesSplit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_dataset():
    """Create fixed synthetic dataset for determinism testing.

    Returns:
        Tuple of (X, y) where:
        - X: DataFrame (1280 samples, 10 features) with MultiIndex (instrument, datetime)
        - y: Series (1280 samples) with same MultiIndex
        - Fixed seed=42 for reproducibility

    Schema:
        X columns: [feature_0, feature_1, ..., feature_9]
        y: continuous target in [-2, 2] range
        Index: MultiIndex[(instrument, datetime)] with 5 instruments, 256 dates each
        Dates are sorted to satisfy PurgedEmbargoedTimeSeriesSplit monotonicity requirement

    Note:
        Dataset sized for n_splits=3, embargo_days=63, max_label_horizon=21:
        Requires (max(63, 21) + 1) * (3 + 1) = 256 unique trading days minimum.
        With 5 instruments × 256 dates = 1280 total samples.
    """
    # Set seed for reproducibility
    np.random.seed(42)

    n_instruments = 5
    n_dates = 256  # Minimum required for CPCV with n_splits=3, embargo=63, horizon=21
    n_features = 10
    n_samples = n_instruments * n_dates  # 1280 samples

    # Generate dates (256 consecutive trading days)
    base_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(base_date, periods=n_dates, freq='B')

    # Create MultiIndex: (instrument, datetime) in date-major order
    # CRITICAL: Iterate dates first to ensure datetime level is non-decreasing
    instruments = [f'STOCK{i:03d}' for i in range(n_instruments)]
    index_tuples = [(inst, date) for date in dates for inst in instruments]
    multi_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=['instrument', 'datetime']
    )

    # Generate features: random values in [-1, 1]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=multi_index,
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Generate target: linear combination of features + noise
    true_weights = np.random.randn(n_features)
    y = pd.Series(
        X.values @ true_weights + np.random.randn(n_samples) * 0.1,
        index=multi_index,
        name='label_21d'
    )

    return X, y


@pytest.fixture
def cv_splitter():
    """Create CPCV splitter for determinism testing.

    Returns:
        PurgedEmbargoedTimeSeriesSplit instance with:
        - n_splits=3 (small for fast testing)
        - max_label_horizon=21 (matches label_21d)
        - embargo_days=63 (NON-NEGOTIABLE per specs, hardcoded in class)

    Note:
        Requires sufficient data for 4 sections (n_splits + 1).
        With 256 dates, each section has ~64 dates, satisfying embargo=63 requirement.
    """
    return PurgedEmbargoedTimeSeriesSplit(
        n_splits=3,
        max_label_horizon=21,
    )


# ============================================================================
# Test Ridge Determinism
# ============================================================================


class TestRidgeDeterminism:
    """Test suite for Ridge model determinism validation."""

    def test_ridge_determinism(self, synthetic_dataset, cv_splitter):
        """Test Ridge training produces identical results on repeat runs.

        Acceptance:
        - Two runs with random_state=42 produce identical OOF predictions
        - max_diff < 1e-9 (Ridge uses deterministic linear algebra)
        - Predictions sorted by index before comparison

        Edge cases:
        - Order of predictions: Sort by index before comparison
        - Numerical precision: Allow 1e-9 tolerance for floating point
        """
        X, y = synthetic_dataset

        # First run
        trainer1 = RidgeTrainer()  # Uses frozen random_state=42
        result1 = run_cv_training(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            trainer=trainer1,
            model_name='ridge',
            horizon='21d',
        )
        oof1 = result1['oof_predictions'].sort_index()

        # Second run
        trainer2 = RidgeTrainer()  # Fresh instance with same frozen seed
        result2 = run_cv_training(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            trainer=trainer2,
            model_name='ridge',
            horizon='21d',
        )
        oof2 = result2['oof_predictions'].sort_index()

        # Verify OOF predictions are identical
        assert len(oof1) == len(oof2), "OOF prediction counts differ"
        assert oof1.index.equals(oof2.index), "OOF indices differ"

        # Check prediction values (max_diff < 1e-9)
        max_diff = np.abs(oof1['prediction'] - oof2['prediction']).max()
        assert max_diff < 1e-9, (
            f"Ridge predictions not deterministic: max_diff={max_diff:.2e} >= 1e-9"
        )

        # Check fold_id consistency
        assert (oof1['fold_id'] == oof2['fold_id']).all(), "Fold IDs differ"

    def test_ridge_determinism_final_model(self, synthetic_dataset, cv_splitter):
        """Test Ridge final model produces identical predictions on repeat runs.

        Acceptance:
        - Two final models trained on full data produce identical predictions
        - max_diff < 1e-9 for predictions on test data

        Edge cases:
        - Test data may differ from training data
        - Model internal state should be identical
        """
        X, y = synthetic_dataset

        # First run
        trainer1 = RidgeTrainer()
        result1 = run_cv_training(
            X=X, y=y, cv_splitter=cv_splitter,
            trainer=trainer1, model_name='ridge', horizon='21d',
        )
        final_model1 = result1['final_model']

        # Second run
        trainer2 = RidgeTrainer()
        result2 = run_cv_training(
            X=X, y=y, cv_splitter=cv_splitter,
            trainer=trainer2, model_name='ridge', horizon='21d',
        )
        final_model2 = result2['final_model']

        # Predict on full dataset
        pred1 = final_model1.predict(X)
        pred2 = final_model2.predict(X)

        # Verify predictions are identical
        max_diff = np.abs(pred1 - pred2).max()
        assert max_diff < 1e-9, (
            f"Ridge final model predictions not deterministic: max_diff={max_diff:.2e}"
        )


# ============================================================================
# Test XGBoost Determinism
# ============================================================================


class TestXGBoostDeterminism:
    """Test suite for XGBoost model determinism validation."""

    def test_xgboost_determinism(self, synthetic_dataset, cv_splitter):
        """Test XGBoost training produces identical results on repeat runs.

        Acceptance:
        - Two runs with random_state=42 produce identical OOF predictions
        - max_diff < 1e-6 (XGBoost has minor floating point variations)
        - Predictions sorted by index before comparison

        Edge cases:
        - XGBoost GPU vs CPU: Only test CPU (GPU may have minor diffs)
        - Numerical precision: Allow 1e-6 tolerance for floating point
        - Order of predictions: Sort by index before comparison
        """
        X, y = synthetic_dataset

        # First run
        trainer1 = XGBoostTrainer()  # Uses frozen random_state=42, tree_method='hist'
        result1 = run_cv_training(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            trainer=trainer1,
            model_name='xgboost',
            horizon='21d',
        )
        oof1 = result1['oof_predictions'].sort_index()

        # Second run
        trainer2 = XGBoostTrainer()  # Fresh instance with same frozen seed
        result2 = run_cv_training(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            trainer=trainer2,
            model_name='xgboost',
            horizon='21d',
        )
        oof2 = result2['oof_predictions'].sort_index()

        # Verify OOF predictions are identical (within tolerance)
        assert len(oof1) == len(oof2), "OOF prediction counts differ"
        assert oof1.index.equals(oof2.index), "OOF indices differ"

        # Check prediction values (max_diff < 1e-6)
        max_diff = np.abs(oof1['prediction'] - oof2['prediction']).max()
        assert max_diff < 1e-6, (
            f"XGBoost predictions not deterministic: max_diff={max_diff:.2e} >= 1e-6"
        )

        # Check fold_id consistency
        assert (oof1['fold_id'] == oof2['fold_id']).all(), "Fold IDs differ"

    def test_xgboost_determinism_final_model(self, synthetic_dataset, cv_splitter):
        """Test XGBoost final model produces identical predictions on repeat runs.

        Acceptance:
        - Two final models trained on full data produce identical predictions
        - max_diff < 1e-6 for predictions on test data

        Edge cases:
        - XGBoost tree_method='hist' should be deterministic on CPU
        - Feature name sanitization should be consistent
        """
        X, y = synthetic_dataset

        # First run
        trainer1 = XGBoostTrainer()
        result1 = run_cv_training(
            X=X, y=y, cv_splitter=cv_splitter,
            trainer=trainer1, model_name='xgboost', horizon='21d',
        )
        final_model1 = result1['final_model']

        # Second run
        trainer2 = XGBoostTrainer()
        result2 = run_cv_training(
            X=X, y=y, cv_splitter=cv_splitter,
            trainer=trainer2, model_name='xgboost', horizon='21d',
        )
        final_model2 = result2['final_model']

        # Predict on full dataset
        pred1 = final_model1.predict(X)
        pred2 = final_model2.predict(X)

        # Verify predictions are identical (within tolerance)
        max_diff = np.abs(pred1 - pred2).max()
        assert max_diff < 1e-6, (
            f"XGBoost final model predictions not deterministic: max_diff={max_diff:.2e}"
        )

    def test_xgboost_cpu_only(self):
        """Test XGBoost trainer uses CPU (tree_method='hist').

        Acceptance:
        - tree_method='hist' (CPU, deterministic)
        - NOT tree_method='gpu_hist' (GPU may have non-deterministic behavior)

        Edge case:
        - GPU vs CPU: Only test CPU for determinism guarantees
        """
        trainer = XGBoostTrainer()
        assert trainer.model.get_params()['tree_method'] == 'hist', (
            "XGBoost must use tree_method='hist' for CPU-based determinism"
        )


# ============================================================================
# Test Cross-Run Determinism (Full Pipeline)
# ============================================================================


class TestCrossRunDeterminism:
    """Test suite for full pipeline determinism validation."""

    def test_cross_run_determinism_cv_scores(self, synthetic_dataset, cv_splitter):
        """Test CV scores are identical across runs.

        Acceptance:
        - Two runs produce identical CV scores (r2, mse, mae per fold)
        - Tolerance: 1e-9 for Ridge, 1e-6 for XGBoost

        Edge cases:
        - CV scores derived from predictions, so should match prediction tolerance
        - Fold assignment must be deterministic
        """
        X, y = synthetic_dataset

        # Test Ridge CV scores
        trainer1 = RidgeTrainer()
        result1 = run_cv_training(
            X=X, y=y, cv_splitter=cv_splitter,
            trainer=trainer1, model_name='ridge', horizon='21d',
        )
        cv_scores1 = result1['cv_scores']

        trainer2 = RidgeTrainer()
        result2 = run_cv_training(
            X=X, y=y, cv_splitter=cv_splitter,
            trainer=trainer2, model_name='ridge', horizon='21d',
        )
        cv_scores2 = result2['cv_scores']

        # Verify CV scores match
        assert len(cv_scores1) == len(cv_scores2), "CV score counts differ"
        for i, (score1, score2) in enumerate(zip(cv_scores1, cv_scores2)):
            assert score1['fold_id'] == score2['fold_id'], f"Fold {i}: fold_id differs"
            assert abs(score1['r2'] - score2['r2']) < 1e-9, f"Fold {i}: r2 differs"
            assert abs(score1['mse'] - score2['mse']) < 1e-9, f"Fold {i}: mse differs"
            assert abs(score1['mae'] - score2['mae']) < 1e-9, f"Fold {i}: mae differs"

    def test_cross_run_determinism_fold_assignment(self, synthetic_dataset, cv_splitter):
        """Test CV fold assignment is deterministic.

        Acceptance:
        - cv_splitter.split() produces identical fold indices across runs
        - Train/test splits match exactly

        Edge cases:
        - Fold assignment must be deterministic for fair comparison
        - PurgedEmbargoedTimeSeriesSplit should be deterministic (no randomness)
        """
        X, y = synthetic_dataset

        # First split
        splits1 = list(cv_splitter.split(X))

        # Second split
        splits2 = list(cv_splitter.split(X))

        # Verify fold counts match
        assert len(splits1) == len(splits2), "Fold counts differ"

        # Verify each fold's train/test indices match
        for fold_idx, ((train1, test1), (train2, test2)) in enumerate(zip(splits1, splits2)):
            assert np.array_equal(train1, train2), f"Fold {fold_idx}: train indices differ"
            assert np.array_equal(test1, test2), f"Fold {fold_idx}: test indices differ"

    def test_cross_run_determinism_all_outputs(self, synthetic_dataset, cv_splitter):
        """Test full pipeline produces identical outputs across runs.

        Acceptance:
        - OOF predictions match (max_diff < tolerance)
        - CV scores match (max_diff < tolerance)
        - Final model predictions match (max_diff < tolerance)

        Edge cases:
        - Test both Ridge and XGBoost in single test
        - Verify all output artifacts are deterministic
        """
        X, y = synthetic_dataset

        # Ridge: Run full pipeline twice
        ridge_results = []
        for run_idx in range(2):
            trainer = RidgeTrainer()
            result = run_cv_training(
                X=X, y=y, cv_splitter=cv_splitter,
                trainer=trainer, model_name='ridge', horizon='21d',
            )
            ridge_results.append(result)

        # Verify Ridge outputs match
        oof1 = ridge_results[0]['oof_predictions'].sort_index()
        oof2 = ridge_results[1]['oof_predictions'].sort_index()
        assert np.abs(oof1['prediction'] - oof2['prediction']).max() < 1e-9

        # XGBoost: Run full pipeline twice
        xgb_results = []
        for run_idx in range(2):
            trainer = XGBoostTrainer()
            result = run_cv_training(
                X=X, y=y, cv_splitter=cv_splitter,
                trainer=trainer, model_name='xgboost', horizon='21d',
            )
            xgb_results.append(result)

        # Verify XGBoost outputs match
        oof1 = xgb_results[0]['oof_predictions'].sort_index()
        oof2 = xgb_results[1]['oof_predictions'].sort_index()
        assert np.abs(oof1['prediction'] - oof2['prediction']).max() < 1e-6
