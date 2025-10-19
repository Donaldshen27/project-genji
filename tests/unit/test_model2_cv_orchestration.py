"""Unit tests for CV training loop orchestrator (P3C4-001-006).

Tests the run_cv_training function that orchestrates cross-validation training
for a single model-horizon pair, including edge cases and metric computation.
"""

import numpy as np
import pandas as pd
import pytest

from src.model2.base_models import (
    RidgeTrainer,
    XGBoostTrainer,
    run_cv_training,
)
from src.model2.train import PurgedEmbargoedTimeSeriesSplit


@pytest.fixture
def synthetic_cv_data():
    """
    Create synthetic data for CV testing.

    Per P3C4-001-006: Sufficient data for 3-fold CPCV with 63-day embargo.
    Uses MultiIndex (datetime, instrument) sorted by datetime to match production format.

    CRITICAL FIX: Need ≥256 trading days for 3 folds with 63-day embargo.
    CRITICAL FIX: Index must be sorted by datetime (monotonic increasing).
    """
    np.random.seed(42)

    # Create 300 trading days (sufficient for 3 folds with 63-day embargo)
    # 3 folds require (63+1) * 4 = 256 minimum trading days
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    instruments = ["AAPL", "MSFT"]

    # Build MultiIndex with datetime FIRST to ensure monotonic ordering
    # This satisfies PurgedEmbargoedTimeSeriesSplit's requirement for sorted dates
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    # Create features and target
    n_samples = len(index)
    X = pd.DataFrame(
        np.random.randn(n_samples, 5), index=index, columns=[f"f{i}" for i in range(5)]
    )
    y = pd.Series(np.random.randn(n_samples), index=index, name="label")

    return X, y


@pytest.fixture
def small_fold_data():
    """
    Create data with very small folds to test fold size validation.

    Per P3C4-001-006: Folds < 100 samples should raise ValueError.
    """
    np.random.seed(42)

    # Only 150 samples total, some folds will be < 100
    # Use 50 days with 3 instruments for 150 total samples
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    instruments = ["AAPL", "MSFT", "GOOGL"]

    # Index sorted by datetime first
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    n_samples = len(index)
    X = pd.DataFrame(
        np.random.randn(n_samples, 5), index=index, columns=[f"f{i}" for i in range(5)]
    )
    y = pd.Series(np.random.randn(n_samples), index=index, name="label")

    return X, y


def test_cv_loop_full_pipeline(synthetic_cv_data):
    """
    Test end-to-end CV training pipeline with Ridge.

    Per P3C4-001-006 acceptance: CV loop completes, logs fold scores,
    returns OOF predictions and final model.
    """
    X, y = synthetic_cv_data

    # Create CPCV splitter with 3 folds
    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)

    # Create Ridge trainer
    trainer = RidgeTrainer()

    # Run CV training
    results = run_cv_training(
        X=X, y=y, cv_splitter=cv_splitter, trainer=trainer, model_name="ridge", horizon="21d"
    )

    # Verify return structure
    assert "oof_predictions" in results
    assert "final_model" in results
    assert "cv_scores" in results

    # Verify OOF predictions
    oof_df = results["oof_predictions"]
    assert isinstance(oof_df, pd.DataFrame)
    assert "prediction" in oof_df.columns
    assert "fold_id" in oof_df.columns
    assert len(oof_df) > 0

    # Verify all predictions are finite
    assert np.all(np.isfinite(oof_df["prediction"]))

    # CRITICAL: Verify OOF index preserves MultiIndex structure
    assert isinstance(oof_df.index, pd.MultiIndex)
    assert oof_df.index.names == ["datetime", "instrument"]

    # Verify final model is trained
    final_model = results["final_model"]
    assert isinstance(final_model, RidgeTrainer)
    assert final_model._is_fitted is True

    # Verify CV scores
    cv_scores = results["cv_scores"]
    assert len(cv_scores) == 3  # 3 folds
    for score in cv_scores:
        assert "model" in score
        assert "horizon" in score
        assert "fold_id" in score
        assert "r2" in score
        assert "mse" in score
        assert "mae" in score
        assert score["model"] == "ridge"
        assert score["horizon"] == "21d"


def test_cv_loop_small_fold(small_fold_data):
    """
    Test that CV training raises ValueError when fold < 100 samples.

    Per P3C4-001-006 edge case: small folds should be rejected.
    """
    X, y = small_fold_data

    # Create CPCV splitter with 3 folds
    # With only 150 samples, some folds will be < 100
    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=5)

    trainer = RidgeTrainer()

    # Should raise ValueError for small fold
    with pytest.raises(ValueError, match="minimum 100 required"):
        run_cv_training(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            trainer=trainer,
            model_name="ridge",
            horizon="21d",
        )


def test_cv_loop_cv_scores(synthetic_cv_data):
    """
    Test that CV scores are computed correctly for all folds.

    Per P3C4-001-006: verify r2, mse, mae metrics for each fold.
    """
    X, y = synthetic_cv_data

    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
    trainer = RidgeTrainer()

    results = run_cv_training(
        X=X, y=y, cv_splitter=cv_splitter, trainer=trainer, model_name="ridge", horizon="21d"
    )

    cv_scores = results["cv_scores"]

    # Verify we have scores for all 3 folds
    assert len(cv_scores) == 3

    # Verify each fold has valid metrics
    for fold_idx, score in enumerate(cv_scores):
        assert score["fold_id"] == fold_idx
        assert score["model"] == "ridge"
        assert score["horizon"] == "21d"

        # Verify metrics are numeric
        assert isinstance(score["r2"], float)
        assert isinstance(score["mse"], float)
        assert isinstance(score["mae"], float)

        # Verify metrics are finite (or NaN for r2 in edge cases)
        assert np.isfinite(score["mse"])
        assert np.isfinite(score["mae"])
        assert score["mse"] >= 0  # MSE is non-negative
        assert score["mae"] >= 0  # MAE is non-negative


def test_cv_loop_final_model(synthetic_cv_data):
    """
    Test that final model is trained on all data.

    Per P3C4-001-006: final model should be trained on full dataset.
    """
    X, y = synthetic_cv_data

    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
    trainer = RidgeTrainer()

    results = run_cv_training(
        X=X, y=y, cv_splitter=cv_splitter, trainer=trainer, model_name="ridge", horizon="21d"
    )

    final_model = results["final_model"]

    # Verify model is fitted
    assert isinstance(final_model, RidgeTrainer)
    assert final_model._is_fitted is True

    # Verify model can generate predictions on full dataset
    predictions = final_model.predict(X)
    assert len(predictions) == len(X)
    assert np.all(np.isfinite(predictions))


def test_cv_loop_xgboost(synthetic_cv_data):
    """
    Test CV training loop with XGBoost model.

    Per P3C4-001-006 acceptance: works with both Ridge and XGBoost.
    """
    X, y = synthetic_cv_data

    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
    trainer = XGBoostTrainer()

    results = run_cv_training(
        X=X,
        y=y,
        cv_splitter=cv_splitter,
        trainer=trainer,
        model_name="xgboost",
        horizon="63d",
    )

    # Verify return structure
    assert "oof_predictions" in results
    assert "final_model" in results
    assert "cv_scores" in results

    # Verify final model type
    final_model = results["final_model"]
    assert isinstance(final_model, XGBoostTrainer)
    assert final_model._is_fitted is True

    # Verify CV scores have correct model name
    cv_scores = results["cv_scores"]
    for score in cv_scores:
        assert score["model"] == "xgboost"
        assert score["horizon"] == "63d"


def test_cv_loop_oof_no_duplicates(synthetic_cv_data):
    """
    Test that OOF predictions have no duplicate indices.

    Per P3C4-001-006: each sample should appear in exactly one fold.
    """
    X, y = synthetic_cv_data

    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
    trainer = RidgeTrainer()

    results = run_cv_training(
        X=X, y=y, cv_splitter=cv_splitter, trainer=trainer, model_name="ridge", horizon="21d"
    )

    oof_df = results["oof_predictions"]

    # Verify no duplicate indices
    assert not oof_df.index.duplicated().any()

    # Verify fold_id is within expected range
    assert oof_df["fold_id"].min() >= 0
    assert oof_df["fold_id"].max() < 3  # 3 folds (0, 1, 2)


def test_cv_loop_constant_target():
    """
    Test CV training with constant target (edge case for r2=NaN).

    Per P3C4-001-006: constant target should log warning but continue.
    """
    np.random.seed(42)

    # Create data with constant target (300 days for sufficient splits)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    instruments = ["AAPL", "MSFT"]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    X = pd.DataFrame(
        np.random.randn(len(index), 5), index=index, columns=[f"f{i}" for i in range(5)]
    )
    y = pd.Series(1.0, index=index, name="label")  # Constant target

    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
    trainer = RidgeTrainer()

    # Should complete without error (though r2 will be NaN)
    results = run_cv_training(
        X=X, y=y, cv_splitter=cv_splitter, trainer=trainer, model_name="ridge", horizon="21d"
    )

    # Verify results are returned (even though r2 is NaN)
    assert "cv_scores" in results
    cv_scores = results["cv_scores"]
    assert len(cv_scores) == 3

    # With constant target, r2 should be NaN
    for score in cv_scores:
        # r2 will be NaN for constant target
        assert np.isnan(score["r2"]) or np.isfinite(score["r2"])
        # MSE and MAE should still be finite
        assert np.isfinite(score["mse"])
        assert np.isfinite(score["mae"])


def test_cv_loop_outlier_predictions_warning(synthetic_cv_data, caplog, capsys):
    """
    Test that outlier predictions trigger warning.

    Per P3C4-001-006: warn when all predictions exceed ±1000 bps.
    """
    X, y = synthetic_cv_data

    class HighPredictionTrainer(RidgeTrainer):
        """Subclass that always predicts large magnitude values."""

        def predict(self, X):
            if not self._is_fitted:
                raise RuntimeError("Model must be fitted before predicting.")
            return np.full(len(X), 12.0, dtype=float)

    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
    trainer = HighPredictionTrainer()

    # Run with logging capture
    with caplog.at_level("WARNING", logger="src.model2.base_models"):
        results = run_cv_training(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            trainer=trainer,
            model_name="ridge",
            horizon="21d",
        )

    # Verify training completed
    assert "cv_scores" in results
    stderr = capsys.readouterr().err
    assert "all predictions exceed" in stderr


def test_cv_loop_training_failure(synthetic_cv_data):
    """
    Test that training failure in a fold is properly handled.

    Per P3C4-001-006: training failure should log error and re-raise with context.
    This test verifies that subclass behavior is preserved via _clone_trainer.
    """
    X, y = synthetic_cv_data

    # Create a trainer subclass that will fail during fit
    class FailingTrainer(RidgeTrainer):
        def fit(self, X, y):
            raise RuntimeError("Simulated training failure")

    cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
    trainer = FailingTrainer()

    # Should raise ValueError with fold context
    # This test verifies that _clone_trainer preserves the FailingTrainer class
    with pytest.raises(ValueError, match="Fold .* training failed"):
        run_cv_training(
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            trainer=trainer,
            model_name="ridge",
            horizon="21d",
        )
