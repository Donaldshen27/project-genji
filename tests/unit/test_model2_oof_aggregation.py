"""Unit tests for OOF prediction aggregation (P3C4-001-005).

Tests the aggregate_oof_predictions function which combines
out-of-fold predictions from multiple cross-validation folds.

Per ticket P3C4-001-005:
- Full coverage verification (all indices present once)
- Duplicate detection (overlapping indices)
- NaN/Inf prediction detection
- Empty fold handling
- Shape mismatch detection
- Invalid input format handling
"""

import numpy as np
import pandas as pd
import pytest

from src.model2.train import aggregate_oof_predictions


def test_oof_aggregation_full_coverage():
    """Verify all indices present exactly once across 5 folds.

    Per P3C4-001-005: test_oof_aggregation_full_coverage requirement.
    Create 5 non-overlapping folds and verify aggregation produces
    complete coverage with correct fold_id assignment.
    """
    # Create 5 non-overlapping folds with 20 samples each
    fold_predictions = [
        (np.arange(0, 20), np.random.randn(20), 0),
        (np.arange(20, 40), np.random.randn(20), 1),
        (np.arange(40, 60), np.random.randn(20), 2),
        (np.arange(60, 80), np.random.randn(20), 3),
        (np.arange(80, 100), np.random.randn(20), 4),
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Verify all indices present
    assert len(result) == 100
    assert set(result.index) == set(range(100))

    # Verify columns
    assert list(result.columns) == ["prediction", "fold_id"]

    # Verify fold_id assignment
    for idx in range(0, 20):
        assert result.loc[idx, "fold_id"] == 0
    for idx in range(20, 40):
        assert result.loc[idx, "fold_id"] == 1
    for idx in range(40, 60):
        assert result.loc[idx, "fold_id"] == 2
    for idx in range(60, 80):
        assert result.loc[idx, "fold_id"] == 3
    for idx in range(80, 100):
        assert result.loc[idx, "fold_id"] == 4

    # Verify predictions are finite
    assert np.all(np.isfinite(result["prediction"]))

    # Verify result is sorted by index
    assert result.index.is_monotonic_increasing


def test_oof_aggregation_duplicate_indices():
    """Verify ValueError raised on overlapping fold indices.

    Per P3C4-001-005: test_oof_aggregation_duplicate_indices requirement.
    Create 2 folds with overlapping indices and verify error message
    contains overlap information.
    """
    # Create 2 folds with overlapping indices (5, 6, 7)
    fold_predictions = [
        (np.arange(0, 10), np.random.randn(10), 0),
        (np.arange(5, 15), np.random.randn(10), 1),  # Overlap: 5, 6, 7, 8, 9
    ]

    with pytest.raises(ValueError, match="overlapping OOF indices"):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_nan_prediction():
    """Verify ValueError raised on NaN predictions.

    Per P3C4-001-005: test_oof_aggregation_nan_prediction requirement.
    Create fold with NaN prediction and verify error message contains
    fold_id information.
    """
    # Create fold with NaN prediction
    predictions_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    fold_predictions = [
        (np.arange(0, 5), predictions_with_nan, 0),
    ]

    with pytest.raises(ValueError, match="Fold 0 contains NaN or infinite predictions"):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_inf_prediction():
    """Verify ValueError raised on infinite predictions.

    Edge case: Predictions contain positive or negative infinity.
    """
    # Create fold with positive infinity
    predictions_with_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
    fold_predictions = [
        (np.arange(0, 5), predictions_with_inf, 0),
    ]

    with pytest.raises(ValueError, match="Fold 0 contains NaN or infinite predictions"):
        aggregate_oof_predictions(fold_predictions)

    # Create fold with negative infinity
    predictions_with_neg_inf = np.array([1.0, 2.0, -np.inf, 4.0, 5.0])
    fold_predictions = [
        (np.arange(0, 5), predictions_with_neg_inf, 1),
    ]

    with pytest.raises(ValueError, match="Fold 1 contains NaN or infinite predictions"):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_empty_fold_list():
    """Verify ValueError raised on empty fold list.

    Edge case per implementation: empty fold_predictions should raise ValueError.
    """
    fold_predictions = []

    with pytest.raises(ValueError, match="fold_predictions cannot be empty"):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_shape_mismatch():
    """Verify ValueError raised when indices and predictions have different lengths.

    Edge case: Shape mismatch between indices and predictions arrays.
    """
    # Indices: 10 elements, Predictions: 5 elements
    fold_predictions = [
        (np.arange(0, 10), np.random.randn(5), 0),
    ]

    with pytest.raises(
        ValueError, match="Fold 0 has mismatched indices \\(10\\) and predictions \\(5\\)"
    ):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_invalid_tuple_format():
    """Verify ValueError raised on invalid tuple format.

    Edge case: Each entry must be a 3-tuple (indices, predictions, fold_id).
    """
    # Only 2 elements in tuple (missing fold_id)
    fold_predictions = [
        (np.arange(0, 10), np.random.randn(10)),  # type: ignore
    ]

    with pytest.raises(
        ValueError,
        match="Each fold prediction entry must be a tuple of \\(indices, predictions, fold_id\\)",
    ):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_non_1d_predictions():
    """Verify ValueError raised when predictions are not 1-dimensional.

    Edge case: Predictions must be 1-D array.
    """
    # 2-D predictions array
    fold_predictions = [
        (np.arange(0, 10), np.random.randn(10, 2), 0),  # 2-D instead of 1-D
    ]

    with pytest.raises(ValueError, match="Predictions for fold 0 must be one-dimensional"):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_string_indices():
    """Verify aggregation works with string indices (e.g., instrument IDs).

    Per implementation: indices can be any hashable type, not just integers.
    This is important for MultiIndex scenarios with (instrument, datetime).
    """
    # Use string indices instead of integers
    fold_predictions = [
        (["AAPL", "GOOGL", "MSFT"], np.array([0.1, 0.2, 0.3]), 0),
        (["TSLA", "AMZN"], np.array([0.4, 0.5]), 1),
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify all indices present
    assert len(result) == 5
    assert set(result.index) == {"AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"}

    # Verify fold_id assignment
    assert result.loc["AAPL", "fold_id"] == 0
    assert result.loc["GOOGL", "fold_id"] == 0
    assert result.loc["MSFT", "fold_id"] == 0
    assert result.loc["TSLA", "fold_id"] == 1
    assert result.loc["AMZN", "fold_id"] == 1


def test_oof_aggregation_tuple_indices():
    """Verify aggregation works with tuple indices (e.g., MultiIndex-like).

    Edge case: indices can be tuples representing (instrument, datetime) pairs.
    """
    # Use tuple indices
    fold_predictions = [
        ([("AAPL", "2020-01-01"), ("AAPL", "2020-01-02")], np.array([0.1, 0.2]), 0),
        ([("GOOGL", "2020-01-01"), ("GOOGL", "2020-01-02")], np.array([0.3, 0.4]), 1),
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify all indices present
    assert len(result) == 4
    expected_indices = {
        ("AAPL", "2020-01-01"),
        ("AAPL", "2020-01-02"),
        ("GOOGL", "2020-01-01"),
        ("GOOGL", "2020-01-02"),
    }
    assert set(result.index) == expected_indices


def test_oof_aggregation_single_fold():
    """Verify aggregation works with single fold.

    Edge case: Only one fold provided (valid for single split).
    """
    fold_predictions = [
        (np.arange(0, 50), np.random.randn(50), 0),
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify single fold aggregated correctly
    assert len(result) == 50
    assert np.all(result["fold_id"] == 0)
    assert list(result.columns) == ["prediction", "fold_id"]


def test_oof_aggregation_preserves_prediction_values():
    """Verify prediction values are preserved during aggregation.

    Test that input prediction values match output prediction values exactly.
    """
    # Create known prediction values
    np.random.seed(42)
    predictions_fold0 = np.random.randn(10)
    predictions_fold1 = np.random.randn(10)

    fold_predictions = [
        (np.arange(0, 10), predictions_fold0, 0),
        (np.arange(10, 20), predictions_fold1, 1),
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify prediction values match
    for idx in range(10):
        assert np.isclose(result.loc[idx, "prediction"], predictions_fold0[idx])
    for idx in range(10, 20):
        assert np.isclose(result.loc[idx, "prediction"], predictions_fold1[idx - 10])


def test_oof_aggregation_fold_id_type():
    """Verify fold_id column has correct dtype (int8).

    Per implementation: fold_id is stored as np.int8 for memory efficiency.
    """
    fold_predictions = [
        (np.arange(0, 10), np.random.randn(10), 0),
        (np.arange(10, 20), np.random.randn(10), 1),  # Fixed: 10 predictions, not 20
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify fold_id dtype is int8
    assert result["fold_id"].dtype == np.int8


def test_oof_aggregation_large_fold_ids():
    """Verify aggregation works with larger fold IDs.

    Edge case: fold_id can be any integer, not limited to 0-4.
    """
    fold_predictions = [
        (np.arange(0, 10), np.random.randn(10), 10),
        (np.arange(10, 20), np.random.randn(10), 20),
        (np.arange(20, 30), np.random.randn(10), 30),
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify fold_id values preserved
    assert set(result["fold_id"].unique()) == {10, 20, 30}
    assert np.all(result.loc[0:9, "fold_id"] == 10)
    assert np.all(result.loc[10:19, "fold_id"] == 20)
    assert np.all(result.loc[20:29, "fold_id"] == 30)


def test_oof_aggregation_duplicate_detection_multiple_folds():
    """Verify duplicate detection works across multiple folds.

    Edge case: Duplicates between non-adjacent folds should be detected.
    """
    # Fold 0 and Fold 2 have overlapping indices (not adjacent)
    fold_predictions = [
        (np.arange(0, 10), np.random.randn(10), 0),
        (np.arange(10, 20), np.random.randn(10), 1),
        (np.arange(5, 15), np.random.randn(10), 2),  # Overlaps with both fold 0 and 1
    ]

    with pytest.raises(ValueError, match="overlapping OOF indices"):
        aggregate_oof_predictions(fold_predictions)


def test_oof_aggregation_predictions_as_list():
    """Verify aggregation works when predictions are provided as lists.

    Per implementation: predictions are converted to np.ndarray via np.asarray.
    """
    fold_predictions = [
        (list(range(0, 10)), [0.1] * 10, 0),  # Lists instead of arrays
        (list(range(10, 20)), [0.2] * 10, 1),
    ]

    result = aggregate_oof_predictions(fold_predictions)

    # Verify aggregation succeeded
    assert len(result) == 20
    assert list(result.columns) == ["prediction", "fold_id"]

    # Verify prediction values
    assert np.allclose(result.loc[0:9, "prediction"], 0.1)
    assert np.allclose(result.loc[10:19, "prediction"], 0.2)
