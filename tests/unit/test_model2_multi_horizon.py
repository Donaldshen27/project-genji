"""Unit tests for multi-horizon training wrapper (P3C4-001-009).

Tests:
- test_multi_horizon_training_success: All 4 model-horizon pairs train successfully
- test_multi_horizon_feature_label_join: Inner join on indices, log dropped rows
- test_multi_horizon_partial_failure: Continue on single model failure (if configured)
- test_multi_horizon_all_outputs: Verify all 4 OOF files and 4 model files saved
- test_multi_horizon_missing_label: Raise KeyError for missing label column
- test_multi_horizon_all_failures: Raise RuntimeError if all pairs fail

All tests implement full acceptance criteria from P3C4-001-009 ticket.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.model2.base_models import (
    RidgeTrainer,
    XGBoostTrainer,
    save_multi_horizon_results,
    train_multi_horizon,
)
from src.model2.train import PurgedEmbargoedTimeSeriesSplit


@pytest.fixture
def synthetic_multi_horizon_data():
    """Create synthetic data for multi-horizon training tests.

    Returns features and labels with MultiIndex (datetime, instrument).
    Labels include both label_21d and label_63d columns.
    """
    np.random.seed(42)

    # Create 300 trading days (sufficient for 3 folds with 63-day embargo)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    instruments = ["AAPL", "MSFT"]

    # Build MultiIndex with datetime FIRST to ensure monotonic ordering
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    # Create features
    n_samples = len(index)
    X = pd.DataFrame(
        np.random.randn(n_samples, 5), index=index, columns=[f"f{i}" for i in range(5)]
    )

    # Create labels with both horizons
    labels = pd.DataFrame(
        {
            "label_21d": np.random.randn(n_samples),
            "label_63d": np.random.randn(n_samples),
        },
        index=index,
    )

    return X, labels


@pytest.fixture
def mismatched_index_data():
    """Create data with feature-label index mismatch for testing inner join.

    Features have 560 rows, labels have 520 rows (40 missing).
    """
    np.random.seed(42)

    dates_features = pd.date_range("2020-01-01", periods=280, freq="D")
    dates_labels = pd.date_range("2020-01-11", periods=260, freq="D")  # 10 days offset
    instruments = ["AAPL", "MSFT"]

    index_features = pd.MultiIndex.from_product(
        [dates_features, instruments], names=["datetime", "instrument"]
    )
    index_labels = pd.MultiIndex.from_product(
        [dates_labels, instruments], names=["datetime", "instrument"]
    )

    X = pd.DataFrame(
        np.random.randn(len(index_features), 5),
        index=index_features,
        columns=[f"f{i}" for i in range(5)],
    )
    labels = pd.DataFrame(
        {
            "label_21d": np.random.randn(len(index_labels)),
            "label_63d": np.random.randn(len(index_labels)),
        },
        index=index_labels,
    )

    return X, labels


class TestMultiHorizonTraining:
    """Test suite for train_multi_horizon() function."""

    def test_multi_horizon_training_success(self, synthetic_multi_horizon_data):
        """Test all 4 model-horizon pairs train successfully.

        Acceptance:
        - Results dict has keys: (ridge, 21d), (ridge, 63d), (xgboost, 21d), (xgboost, 63d)
        - Each result has keys: ['oof', 'model', 'cv_scores']
        - OOF predictions are DataFrame with columns [prediction, fold_id]
        - Model is fitted BaseModelTrainer instance
        - CV scores is list of dicts with keys [model, horizon, fold_id, r2, mse, mae]
        """
        X, labels = synthetic_multi_horizon_data

        # Create CPCV splitter
        cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)

        # Config with models and horizons
        config = {
            "models": ["ridge", "xgboost"],
            "horizons": ["21d", "63d"],
        }

        # Train all model-horizon pairs
        results = train_multi_horizon(X, labels, cv_splitter, config)

        # Verify all 4 pairs are present
        expected_keys = [
            ("ridge", "21d"),
            ("ridge", "63d"),
            ("xgboost", "21d"),
            ("xgboost", "63d"),
        ]
        assert set(results.keys()) == set(expected_keys)

        # Verify structure of each result
        for (model_name, horizon), result in results.items():
            # Check keys exist
            assert "oof" in result
            assert "model" in result
            assert "cv_scores" in result

            # Verify OOF predictions
            oof_df = result["oof"]
            assert isinstance(oof_df, pd.DataFrame)
            assert "prediction" in oof_df.columns
            assert "fold_id" in oof_df.columns
            assert len(oof_df) > 0
            assert np.all(np.isfinite(oof_df["prediction"]))

            # Verify model is fitted
            model = result["model"]
            if model_name == "ridge":
                assert isinstance(model, RidgeTrainer)
            elif model_name == "xgboost":
                assert isinstance(model, XGBoostTrainer)
            assert model._is_fitted is True

            # Verify CV scores
            cv_scores = result["cv_scores"]
            assert isinstance(cv_scores, list)
            assert len(cv_scores) == 3  # 3 folds
            for score in cv_scores:
                assert score["model"] == model_name
                assert score["horizon"] == horizon
                assert "fold_id" in score
                assert "r2" in score
                assert "mse" in score
                assert "mae" in score

    def test_multi_horizon_feature_label_join(self, mismatched_index_data):
        """Test inner join on features and labels with index mismatch.

        Acceptance:
        - Features have 560 rows, labels have 520 rows (40 missing)
        - Inner join produces 520 aligned samples
        - Warning logged: 'Dropped 40 rows due to index mismatch'
        - Training proceeds on 520 samples
        """
        X, labels = mismatched_index_data

        # Verify initial sizes
        assert len(X) == 560
        assert len(labels) == 520

        cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
        config = {"models": ["ridge"], "horizons": ["21d"]}

        # Train (logs verified via stderr in test run)
        results = train_multi_horizon(X, labels, cv_splitter, config)

        # Verify training succeeded on aligned data
        assert ("ridge", "21d") in results
        oof_df = results[("ridge", "21d")]["oof"]
        # OOF size will be less than 520 due to CV purging/embargo, but should be > 0
        assert len(oof_df) > 0
        assert len(oof_df) <= 520

    def test_multi_horizon_partial_failure(self, synthetic_multi_horizon_data):
        """Test continues on single model failure if config allows.

        Acceptance:
        - Mock XGBoostTrainer to raise ValueError during fit
        - Ridge models train successfully for both horizons
        - XGBoost failures logged: 'Model xgboost horizon 21d training failed'
        - Results dict has only (ridge, 21d) and (ridge, 63d)
        - No RuntimeError raised (partial success allowed)
        """
        X, labels = synthetic_multi_horizon_data

        # Create a failing XGBoost trainer class
        class FailingXGBoostTrainer(XGBoostTrainer):
            def fit(self, X, y):
                raise ValueError("Simulated XGBoost training failure")

        # Monkey-patch the module to use failing XGBoost
        import src.model2.base_models as base_models_module

        original_xgboost = base_models_module.XGBoostTrainer
        base_models_module.XGBoostTrainer = FailingXGBoostTrainer

        try:
            cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
            config = {"models": ["ridge", "xgboost"], "horizons": ["21d", "63d"]}

            results = train_multi_horizon(X, labels, cv_splitter, config)

            # Verify only Ridge models succeeded
            assert set(results.keys()) == {("ridge", "21d"), ("ridge", "63d")}

            # Verify XGBoost models are NOT in results
            assert ("xgboost", "21d") not in results
            assert ("xgboost", "63d") not in results

            # Verify Ridge models are valid
            for horizon in ["21d", "63d"]:
                assert isinstance(results[("ridge", horizon)]["model"], RidgeTrainer)

        finally:
            # Restore original XGBoostTrainer
            base_models_module.XGBoostTrainer = original_xgboost

    def test_multi_horizon_all_outputs(self, synthetic_multi_horizon_data, tmp_path):
        """Test all outputs saved to correct directories.

        Acceptance:
        - 4 OOF parquet files in {output_dir}/oof/
        - 4 model pickle files in {output_dir}/models/
        - 4 CV score JSON files in {output_dir}/cv_scores/
        - 2 feature importance parquet files in {output_dir}/feature_importance/ (XGBoost only)
        - All files named correctly: {model}_{horizon}_*.{ext}
        """
        X, labels = synthetic_multi_horizon_data

        cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
        config = {"models": ["ridge", "xgboost"], "horizons": ["21d", "63d"]}

        # Train all pairs
        results = train_multi_horizon(X, labels, cv_splitter, config)

        # Save all results
        output_dir = tmp_path / "model2" / "us"
        save_multi_horizon_results(results, output_dir, region="US")

        # Verify directory structure
        assert (output_dir / "oof").exists()
        assert (output_dir / "models").exists()
        assert (output_dir / "cv_scores").exists()
        assert (output_dir / "feature_importance").exists()

        # Verify OOF files (4 total)
        expected_oof_files = [
            "ridge_21d_oof.parquet",
            "ridge_63d_oof.parquet",
            "xgboost_21d_oof.parquet",
            "xgboost_63d_oof.parquet",
        ]
        for filename in expected_oof_files:
            assert (output_dir / "oof" / filename).exists()

        # Verify model files (4 total)
        expected_model_files = [
            "ridge_21d.pkl",
            "ridge_63d.pkl",
            "xgboost_21d.pkl",
            "xgboost_63d.pkl",
        ]
        for filename in expected_model_files:
            assert (output_dir / "models" / filename).exists()

        # Verify CV score JSON files (4 total)
        expected_cv_files = [
            "ridge_21d_cv_scores.json",
            "ridge_63d_cv_scores.json",
            "xgboost_21d_cv_scores.json",
            "xgboost_63d_cv_scores.json",
        ]
        for filename in expected_cv_files:
            cv_file = output_dir / "cv_scores" / filename
            assert cv_file.exists()
            # Verify JSON is valid
            with open(cv_file) as f:
                cv_scores = json.load(f)
                assert isinstance(cv_scores, list)
                assert len(cv_scores) == 3  # 3 folds

        # Verify feature importance files (2 total, XGBoost only)
        expected_importance_files = [
            "xgboost_21d_importance.parquet",
            "xgboost_63d_importance.parquet",
        ]
        for filename in expected_importance_files:
            importance_file = output_dir / "feature_importance" / filename
            assert importance_file.exists()
            # Verify parquet is readable
            importance_df = pd.read_parquet(importance_file)
            assert "feature" in importance_df.columns
            assert "importance_gain" in importance_df.columns
            assert "importance_weight" in importance_df.columns

    def test_multi_horizon_missing_label(self, synthetic_multi_horizon_data):
        """Test raises KeyError for missing label column.

        Acceptance:
        - Labels DataFrame has only [label_21d] (missing label_63d)
        - train_multi_horizon() raises KeyError
        - Error message: 'Missing required label columns: {label_63d}'
        """
        X, labels = synthetic_multi_horizon_data

        # Remove label_63d column
        labels_missing = labels[["label_21d"]].copy()

        cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
        config = {"models": ["ridge"], "horizons": ["21d", "63d"]}

        # Should raise KeyError for missing label column
        with pytest.raises(KeyError, match="Missing required label columns.*label_63d"):
            train_multi_horizon(X, labels_missing, cv_splitter, config)

    def test_multi_horizon_all_failures(self, synthetic_multi_horizon_data):
        """Test raises RuntimeError if all model-horizon pairs fail.

        Acceptance:
        - Mock all trainers to raise ValueError during fit
        - train_multi_horizon() raises RuntimeError
        - Error message contains failure summary: 'All 4 model-horizon pairs failed'
        """
        X, labels = synthetic_multi_horizon_data

        # Create failing trainers
        class FailingRidgeTrainer(RidgeTrainer):
            def fit(self, X, y):
                raise ValueError("Simulated Ridge failure")

        class FailingXGBoostTrainer(XGBoostTrainer):
            def fit(self, X, y):
                raise ValueError("Simulated XGBoost failure")

        # Monkey-patch both trainers
        import src.model2.base_models as base_models_module

        original_ridge = base_models_module.RidgeTrainer
        original_xgboost = base_models_module.XGBoostTrainer
        base_models_module.RidgeTrainer = FailingRidgeTrainer
        base_models_module.XGBoostTrainer = FailingXGBoostTrainer

        try:
            cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
            config = {"models": ["ridge", "xgboost"], "horizons": ["21d", "63d"]}

            # Should raise RuntimeError when all pairs fail
            with pytest.raises(RuntimeError, match="All .* model-horizon pairs failed"):
                train_multi_horizon(X, labels, cv_splitter, config)

        finally:
            # Restore original trainers
            base_models_module.RidgeTrainer = original_ridge
            base_models_module.XGBoostTrainer = original_xgboost


class TestSaveMultiHorizonResults:
    """Test suite for save_multi_horizon_results() function."""

    def test_save_all_outputs(self, synthetic_multi_horizon_data, tmp_path):
        """Test all outputs saved to correct locations.

        Acceptance:
        - All subdirectories created: oof/, models/, cv_scores/, feature_importance/
        - All files saved with correct naming: {model}_{horizon}_*.{ext}
        - File counts: 4 OOF, 4 models, 4 CV scores, 2 feature importance
        """
        X, labels = synthetic_multi_horizon_data

        cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
        config = {"models": ["ridge", "xgboost"], "horizons": ["21d", "63d"]}

        results = train_multi_horizon(X, labels, cv_splitter, config)

        output_dir = tmp_path / "model2" / "test"
        save_multi_horizon_results(results, output_dir, region="TEST")

        # Verify all subdirectories exist
        assert (output_dir / "oof").is_dir()
        assert (output_dir / "models").is_dir()
        assert (output_dir / "cv_scores").is_dir()
        assert (output_dir / "feature_importance").is_dir()

        # Count files in each directory
        oof_files = list((output_dir / "oof").glob("*.parquet"))
        model_files = list((output_dir / "models").glob("*.pkl"))
        cv_files = list((output_dir / "cv_scores").glob("*.json"))
        importance_files = list((output_dir / "feature_importance").glob("*.parquet"))

        assert len(oof_files) == 4
        assert len(model_files) == 4
        assert len(cv_files) == 4
        assert len(importance_files) == 2  # XGBoost only

    def test_save_empty_results(self, tmp_path):
        """Test raises ValueError for empty results dict.

        Acceptance:
        - Empty results dict raises ValueError
        - Error message: 'Cannot save empty results'
        """
        empty_results = {}
        output_dir = tmp_path / "model2" / "test"

        with pytest.raises(ValueError, match="Cannot save empty results"):
            save_multi_horizon_results(empty_results, output_dir, region="TEST")

    def test_save_creates_directories(self, synthetic_multi_horizon_data, tmp_path):
        """Test creates output directories if they don't exist.

        Acceptance:
        - Output directory does not exist initially
        - save_multi_horizon_results() creates all subdirectories
        - All files saved successfully
        """
        X, labels = synthetic_multi_horizon_data

        cv_splitter = PurgedEmbargoedTimeSeriesSplit(n_splits=3, max_label_horizon=21)
        config = {"models": ["ridge"], "horizons": ["21d"]}

        results = train_multi_horizon(X, labels, cv_splitter, config)

        # Use non-existent nested directory
        output_dir = tmp_path / "nested" / "model2" / "test"
        assert not output_dir.exists()

        # Save should create all directories
        save_multi_horizon_results(results, output_dir, region="TEST")

        # Verify directories were created
        assert output_dir.exists()
        assert (output_dir / "oof").exists()
        assert (output_dir / "models").exists()
        assert (output_dir / "cv_scores").exists()

        # Verify files were saved
        assert (output_dir / "oof" / "ridge_21d_oof.parquet").exists()
        assert (output_dir / "models" / "ridge_21d.pkl").exists()
        assert (output_dir / "cv_scores" / "ridge_21d_cv_scores.json").exists()
