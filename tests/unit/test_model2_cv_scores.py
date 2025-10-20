"""Unit tests for CV score logging and aggregation (P3C4-001-010).

Tests:
- test_compute_cv_scores_normal: Verify r2, mse, mae computed correctly on toy data
- test_compute_cv_scores_constant_target: Verify r2=NaN logged with warning
- test_compute_cv_scores_nan_predictions: Verify ValueError raised for NaN predictions
- test_compute_cv_scores_negative_r2: Verify negative r2 logged with warning
- test_validate_cv_score_schema_valid: Verify valid schema passes
- test_validate_cv_score_schema_missing_keys: Verify ValueError for missing keys
- test_validate_cv_score_schema_invalid_types: Verify ValueError for wrong types
- test_aggregate_cv_scores_normal: Verify mean and std computed across folds
- test_aggregate_cv_scores_with_nan: Verify np.nanmean/np.nanstd handle NaN
- test_aggregate_cv_scores_empty: Verify ValueError for empty list
- test_log_cv_scores_json: Verify JSON output with allow_nan=True
- test_log_cv_scores_json_invalid_schema: Verify ValueError for schema violation
- test_outlier_threshold_constant: Verify OUTLIER_THRESHOLD_BPS used instead of hardcoded 5.0
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model2.base_models import (
    OUTLIER_THRESHOLD_BPS,
    CV_SCORE_SCHEMA_KEYS,
    aggregate_cv_scores,
    compute_cv_scores,
    log_cv_scores_json,
    validate_cv_score_schema,
    _check_outlier_predictions,
)


class TestComputeCVScores:
    """Test suite for compute_cv_scores() function."""

    def test_compute_cv_scores_normal(self):
        """Test CV score computation on normal data.

        Acceptance:
        - Schema: {model, horizon, fold_id, r2, mse, mae}
        - All metrics are finite floats
        - r2 ∈ [-∞, 1], mse ≥ 0, mae ≥ 0
        """
        np.random.seed(42)

        # Create synthetic data
        y_true = pd.Series(np.random.randn(100))
        y_pred = y_true + np.random.randn(100) * 0.1  # Add small noise

        # Compute CV scores
        scores = compute_cv_scores(
            y_true=y_true,
            y_pred=y_pred,
            fold_idx=0,
            model_name="ridge",
            horizon="21d",
        )

        # Verify schema
        assert set(scores.keys()) == CV_SCORE_SCHEMA_KEYS
        assert scores["model"] == "ridge"
        assert scores["horizon"] == "21d"
        assert scores["fold_id"] == 0

        # Verify metrics are finite
        assert np.isfinite(scores["r2"])
        assert np.isfinite(scores["mse"])
        assert np.isfinite(scores["mae"])

        # Verify metric bounds
        assert scores["r2"] <= 1.0  # r2 can be negative but max is 1.0
        assert scores["mse"] >= 0.0
        assert scores["mae"] >= 0.0

        # With low noise, r2 should be high
        assert scores["r2"] > 0.8

    def test_compute_cv_scores_constant_target(self, caplog):
        """Test CV score computation with constant target.

        Acceptance:
        - r2 is 0.0 or NaN (sklearn may return either for constant target)
        - Warning logged: 'constant target detected'
        - mse and mae are finite
        """
        import logging

        # Create constant target
        np.random.seed(42)
        y_true = pd.Series(np.ones(100) * 5.0)
        y_pred = np.random.randn(100)  # Random predictions

        # Configure logger to capture warnings (pattern from test_model2_base_trainers.py)
        test_logger = logging.getLogger("src.model2.base_models")
        previous_level = test_logger.level
        previous_propagate = test_logger.propagate
        test_logger.setLevel(logging.WARNING)
        test_logger.addHandler(caplog.handler)
        test_logger.propagate = False
        caplog.set_level(logging.WARNING)

        try:
            with caplog.at_level(logging.WARNING, logger="src.model2.base_models"):
                scores = compute_cv_scores(
                    y_true=y_true,
                    y_pred=y_pred,
                    fold_idx=0,
                    model_name="ridge",
                    horizon="21d",
                )
        finally:
            test_logger.propagate = previous_propagate
            test_logger.removeHandler(caplog.handler)
            test_logger.setLevel(previous_level)

        # Verify r2 is 0.0 or NaN (sklearn behavior varies by version)
        assert scores["r2"] == 0.0 or np.isnan(scores["r2"])

        # Verify mse and mae are finite
        assert np.isfinite(scores["mse"])
        assert np.isfinite(scores["mae"])

        # Verify warning logged
        assert "constant target detected" in caplog.text

    def test_compute_cv_scores_nan_predictions(self):
        """Test CV score computation raises ValueError for NaN predictions.

        Acceptance:
        - ValueError raised before metric computation
        - Error message: 'predictions contain NaN or Inf values'
        """
        y_true = pd.Series(np.random.randn(100))
        y_pred = np.random.randn(100)
        y_pred[50] = np.nan  # Inject NaN

        # Should raise ValueError
        with pytest.raises(ValueError, match="predictions contain NaN or Inf"):
            compute_cv_scores(
                y_true=y_true,
                y_pred=y_pred,
                fold_idx=0,
                model_name="ridge",
                horizon="21d",
            )

    def test_compute_cv_scores_negative_r2(self, caplog):
        """Test CV score computation logs warning for negative r2.

        Acceptance:
        - r2 < 0 (model worse than mean baseline) - DETERMINISTIC
        - Warning logged: 'negative r2'
        - r2 value is preserved (not clipped to 0)
        """
        import logging

        # Create data that GUARANTEES negative r2 (deterministic per Codex review)
        np.random.seed(42)
        y_true = pd.Series(np.random.randn(100))
        # Predictions are negatively correlated with true values
        y_pred = -y_true.values + np.random.randn(100) * 0.1

        # Configure logger to capture warnings
        test_logger = logging.getLogger("src.model2.base_models")
        previous_level = test_logger.level
        previous_propagate = test_logger.propagate
        test_logger.setLevel(logging.WARNING)
        test_logger.addHandler(caplog.handler)
        test_logger.propagate = False
        caplog.set_level(logging.WARNING)

        try:
            with caplog.at_level(logging.WARNING, logger="src.model2.base_models"):
                scores = compute_cv_scores(
                    y_true=y_true,
                    y_pred=y_pred,
                    fold_idx=0,
                    model_name="ridge",
                    horizon="21d",
                )
        finally:
            test_logger.propagate = previous_propagate
            test_logger.removeHandler(caplog.handler)
            test_logger.setLevel(previous_level)

        # Verify r2 is negative (UNCONDITIONAL - per Codex review)
        assert scores["r2"] < 0, f"Expected negative r2, got {scores['r2']}"

        # Verify warning logged (UNCONDITIONAL)
        assert "negative r2" in caplog.text

        # Verify value is preserved (not clipped)
        assert np.isfinite(scores["r2"])


class TestValidateCVScoreSchema:
    """Test suite for validate_cv_score_schema() function."""

    def test_validate_cv_score_schema_valid(self):
        """Test schema validation passes for valid CV score dict."""
        valid_score = {
            "model": "ridge",
            "horizon": "21d",
            "fold_id": 0,
            "r2": 0.85,
            "mse": 0.12,
            "mae": 0.28,
        }

        # Should not raise
        validate_cv_score_schema(valid_score)

    def test_validate_cv_score_schema_missing_keys(self):
        """Test schema validation raises ValueError for missing keys."""
        incomplete_score = {
            "model": "ridge",
            "horizon": "21d",
            # Missing: fold_id, r2, mse, mae
        }

        with pytest.raises(ValueError, match="missing required keys"):
            validate_cv_score_schema(incomplete_score)

    def test_validate_cv_score_schema_invalid_types(self):
        """Test schema validation raises ValueError for wrong types."""
        # Test invalid model type (should be str)
        invalid_model_type = {
            "model": 123,  # Should be str
            "horizon": "21d",
            "fold_id": 0,
            "r2": 0.85,
            "mse": 0.12,
            "mae": 0.28,
        }
        with pytest.raises(ValueError, match="'model' must be str"):
            validate_cv_score_schema(invalid_model_type)

        # Test invalid fold_id type (should be int)
        invalid_fold_id_type = {
            "model": "ridge",
            "horizon": "21d",
            "fold_id": "0",  # Should be int
            "r2": 0.85,
            "mse": 0.12,
            "mae": 0.28,
        }
        with pytest.raises(ValueError, match="'fold_id' must be int"):
            validate_cv_score_schema(invalid_fold_id_type)


class TestAggregateCVScores:
    """Test suite for aggregate_cv_scores() function."""

    def test_aggregate_cv_scores_normal(self):
        """Test CV score aggregation on normal data.

        Acceptance:
        - Aggregate schema: {n_folds, r2_mean, r2_std, mse_mean, mse_std, mae_mean, mae_std}
        - Mean and std computed correctly
        """
        cv_scores = [
            {"model": "ridge", "horizon": "21d", "fold_id": 0, "r2": 0.8, "mse": 0.1, "mae": 0.2},
            {"model": "ridge", "horizon": "21d", "fold_id": 1, "r2": 0.85, "mse": 0.09, "mae": 0.19},
            {"model": "ridge", "horizon": "21d", "fold_id": 2, "r2": 0.9, "mse": 0.08, "mae": 0.18},
        ]

        agg_scores = aggregate_cv_scores(cv_scores)

        # Verify schema
        assert "n_folds" in agg_scores
        assert "r2_mean" in agg_scores
        assert "r2_std" in agg_scores
        assert "mse_mean" in agg_scores
        assert "mse_std" in agg_scores
        assert "mae_mean" in agg_scores
        assert "mae_std" in agg_scores

        # Verify values
        assert agg_scores["n_folds"] == 3
        assert np.isclose(agg_scores["r2_mean"], 0.85)
        assert np.isclose(agg_scores["mse_mean"], 0.09)
        assert np.isclose(agg_scores["mae_mean"], 0.19)

        # Verify std is positive
        assert agg_scores["r2_std"] > 0
        assert agg_scores["mse_std"] > 0
        assert agg_scores["mae_std"] > 0

    def test_aggregate_cv_scores_with_nan(self):
        """Test CV score aggregation handles NaN values.

        Acceptance:
        - Uses np.nanmean/np.nanstd to ignore NaN
        - Aggregate computed from non-NaN values
        """
        cv_scores = [
            {"model": "ridge", "horizon": "21d", "fold_id": 0, "r2": np.nan, "mse": 0.1, "mae": 0.2},
            {"model": "ridge", "horizon": "21d", "fold_id": 1, "r2": 0.85, "mse": 0.09, "mae": 0.19},
            {"model": "ridge", "horizon": "21d", "fold_id": 2, "r2": 0.9, "mse": 0.08, "mae": 0.18},
        ]

        agg_scores = aggregate_cv_scores(cv_scores)

        # Verify r2_mean computed from non-NaN values (0.85, 0.9)
        assert agg_scores["n_folds"] == 3
        assert np.isclose(agg_scores["r2_mean"], 0.875)  # Mean of 0.85 and 0.9

    def test_aggregate_cv_scores_empty(self):
        """Test CV score aggregation raises ValueError for empty list."""
        with pytest.raises(ValueError, match="Cannot aggregate empty cv_scores"):
            aggregate_cv_scores([])


class TestLogCVScoresJSON:
    """Test suite for log_cv_scores_json() function."""

    def test_log_cv_scores_json(self):
        """Test CV scores written to JSON with allow_nan=True.

        Acceptance:
        - JSON file created at output_path
        - Contains all CV scores
        - allow_nan=True enables NaN serialization
        """
        cv_scores = [
            {"model": "ridge", "horizon": "21d", "fold_id": 0, "r2": 0.8, "mse": 0.1, "mae": 0.2},
            {"model": "ridge", "horizon": "21d", "fold_id": 1, "r2": np.nan, "mse": 0.09, "mae": 0.19},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cv_scores" / "test_cv_scores.json"

            # Write CV scores
            log_cv_scores_json(cv_scores, output_path)

            # Verify file exists
            assert output_path.exists()

            # Load and verify contents
            with open(output_path) as f:
                loaded_scores = json.load(f)

            assert len(loaded_scores) == 2
            assert loaded_scores[0]["fold_id"] == 0
            assert loaded_scores[0]["r2"] == 0.8

            # NaN should be serialized as NaN (not null)
            # In JSON, NaN is represented as NaN (not a standard JSON value but allowed with allow_nan=True)
            assert np.isnan(loaded_scores[1]["r2"])

    def test_log_cv_scores_json_invalid_schema(self):
        """Test log_cv_scores_json raises ValueError for schema violation."""
        invalid_scores = [
            {"model": "ridge", "horizon": "21d", "fold_id": 0},  # Missing r2, mse, mae
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cv_scores.json"

            with pytest.raises(ValueError, match="missing required keys"):
                log_cv_scores_json(invalid_scores, output_path)

    def test_log_cv_scores_json_empty(self):
        """Test log_cv_scores_json raises ValueError for empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "cv_scores.json"

            with pytest.raises(ValueError, match="Cannot write empty cv_scores"):
                log_cv_scores_json([], output_path)


class TestOutlierThresholdConstant:
    """Test suite for OUTLIER_THRESHOLD_BPS constant."""

    def test_outlier_threshold_constant_value(self):
        """Test OUTLIER_THRESHOLD_BPS has correct value.

        Acceptance:
        - OUTLIER_THRESHOLD_BPS = 5.0 (±500 bps)
        """
        assert OUTLIER_THRESHOLD_BPS == 5.0

    def test_outlier_threshold_used_in_check(self, caplog):
        """Test _check_outlier_predictions uses OUTLIER_THRESHOLD_BPS.

        Acceptance:
        - Warning logged when predictions exceed ±OUTLIER_THRESHOLD_BPS
        - Warning message includes threshold value
        """
        import logging

        # Create predictions that exceed threshold
        predictions = np.array([6.0, 7.0, -6.0, -7.0] * 10)  # 40 predictions, all exceed ±5.0

        # Configure logger to capture warnings
        test_logger = logging.getLogger("src.model2.base_models")
        previous_level = test_logger.level
        previous_propagate = test_logger.propagate
        test_logger.setLevel(logging.WARNING)
        test_logger.addHandler(caplog.handler)
        test_logger.propagate = False
        caplog.set_level(logging.WARNING)

        try:
            with caplog.at_level(logging.WARNING, logger="src.model2.base_models"):
                _check_outlier_predictions(predictions, fold_idx=0)
        finally:
            test_logger.propagate = previous_propagate
            test_logger.removeHandler(caplog.handler)
            test_logger.setLevel(previous_level)

        # Verify warning logged with threshold value
        assert "all predictions exceed ±5.0" in caplog.text or "100.00% of predictions exceed ±5.0" in caplog.text
