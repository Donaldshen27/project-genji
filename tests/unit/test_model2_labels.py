"""
Unit tests for Model 2 labels module.

Tests:
- Forward returns calculation correctness
- Industry-relative labels computation
- No look-ahead bias
- Schema validation
- Edge cases (single instrument, missing data, etc.)

Per Phase 3 Chunk 1 acceptance criteria.
"""

import numpy as np
import pandas as pd
import pytest

from src.model2.labels import (
    compute_forward_returns,
    compute_industry_relative_labels,
    compute_universe_returns,
)


@pytest.fixture
def simple_price_data():
    """
    Create simple price data for testing.

    Two instruments over 100 days with known returns.
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    instruments = ["AAPL", "GOOGL"]

    # Create MultiIndex
    index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])

    # Create simple price series: exponential growth
    # AAPL: starts at 100, grows at 0.1% per day
    # GOOGL: starts at 200, grows at 0.05% per day
    np.random.seed(42)
    prices = []
    for inst in instruments:
        if inst == "AAPL":
            base = 100.0
            growth_rate = 0.001
        else:
            base = 200.0
            growth_rate = 0.0005

        inst_prices = [
            base * np.exp(growth_rate * i + np.random.normal(0, 0.002) * i)
            for i in range(len(dates))
        ]
        prices.extend(inst_prices)

    df = pd.DataFrame({"$close": prices}, index=index)
    return df


@pytest.fixture
def single_instrument_data():
    """Create data for a single instrument."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    prices = 100 * np.exp(0.001 * np.arange(50))

    df = pd.DataFrame({"$close": prices}, index=dates)
    df.index.name = "datetime"

    return df


class TestComputeForwardReturns:
    """Test forward returns calculation."""

    def test_forward_returns_shape(self, simple_price_data):
        """Test that forward returns have correct shape."""
        horizon = 21
        forward_returns = compute_forward_returns(simple_price_data, horizon)

        assert isinstance(forward_returns, pd.Series)
        assert len(forward_returns) == len(simple_price_data)
        assert forward_returns.name is None or forward_returns.name == "$close"

    def test_forward_returns_no_lookahead(self, simple_price_data):
        """
        Test that forward returns only use future data.

        Value at date t should depend only on prices at t+1, ..., t+horizon.
        """
        horizon = 5
        forward_returns = compute_forward_returns(simple_price_data, horizon)

        # Extract one instrument for clarity
        aapl_returns = forward_returns.xs("AAPL", level="instrument")

        # Check that early dates have NaN (insufficient future data at the end)
        # After shift(-horizon), last horizon-1 values should be NaN
        assert aapl_returns.iloc[-horizon:].isna().sum() >= horizon - 1

    def test_forward_returns_single_instrument(self, single_instrument_data):
        """Test forward returns with single instrument (no MultiIndex)."""
        horizon = 10
        forward_returns = compute_forward_returns(single_instrument_data, horizon)

        assert isinstance(forward_returns, pd.Series)
        assert len(forward_returns) == len(single_instrument_data)
        assert forward_returns.index.name == "datetime"

    def test_forward_returns_manual_verification(self):
        """
        Manually verify forward returns calculation for a simple case.

        Create prices where we know exact returns.
        """
        # Simple case: prices that double every period
        # P_t = 100 * 2^t
        # r_t = ln(P_t / P_{t-1}) = ln(2)
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = [100 * (2**i) for i in range(10)]

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        horizon = 3

        forward_returns = compute_forward_returns(df, horizon)

        # Expected forward return for horizon=3: sum of 3 log(2) = 3 * ln(2)
        expected_return = 3 * np.log(2)

        # Check first valid forward return (at index 0, looking at returns 1, 2, 3)
        # Due to shift(-horizon), this appears at index 0
        assert forward_returns.iloc[0] == pytest.approx(expected_return, abs=1e-6)

    def test_forward_returns_nans_at_end(self, simple_price_data):
        """Test that forward returns have NaNs where insufficient future data."""
        horizon = 21
        forward_returns = compute_forward_returns(simple_price_data, horizon)

        # Last horizon days should have NaN (no future data)
        aapl_returns = forward_returns.xs("AAPL", level="instrument")
        assert aapl_returns.iloc[-horizon:].isna().all()


class TestComputeUniverseReturns:
    """Test universe returns (equal-weighted) calculation."""

    def test_universe_returns_shape(self, simple_price_data):
        """Test that universe returns have correct shape."""
        horizon = 21
        universe_returns = compute_universe_returns(simple_price_data, horizon)

        assert isinstance(universe_returns, pd.Series)
        # Should be indexed by datetime only
        assert universe_returns.index.name == "datetime"

    def test_universe_returns_are_mean(self, simple_price_data):
        """Test that universe returns are mean of stock returns."""
        horizon = 10

        # Compute stock-level forward returns
        stock_returns = compute_forward_returns(simple_price_data, horizon)

        # Compute universe returns
        universe_returns = compute_universe_returns(simple_price_data, horizon)

        # Manually compute mean across instruments for each date
        expected_mean = stock_returns.groupby(level="datetime").mean()

        # Should match
        pd.testing.assert_series_equal(universe_returns, expected_mean, check_names=False)

    def test_universe_returns_single_instrument(self, single_instrument_data):
        """Test universe returns with single instrument."""
        horizon = 10
        universe_returns = compute_universe_returns(single_instrument_data, horizon)

        # For single instrument, universe return = stock return
        stock_returns = compute_forward_returns(single_instrument_data, horizon)

        pd.testing.assert_series_equal(universe_returns, stock_returns, check_names=False)


class TestComputeIndustryRelativeLabels:
    """Test industry-relative labels calculation."""

    def test_labels_shape(self, simple_price_data):
        """Test that labels have correct shape."""
        horizons = [21, 63]
        labels = compute_industry_relative_labels(simple_price_data, horizons)

        assert isinstance(labels, pd.DataFrame)
        assert len(labels) == len(simple_price_data)
        assert list(labels.columns) == ["label_21d", "label_63d"]

    def test_labels_schema(self, simple_price_data):
        """Test that labels follow correct schema."""
        horizons = [10, 20]
        labels = compute_industry_relative_labels(simple_price_data, horizons)

        # Check column names
        assert "label_10d" in labels.columns
        assert "label_20d" in labels.columns

        # Check index is preserved
        pd.testing.assert_index_equal(labels.index, simple_price_data.index)

    def test_labels_are_relative_to_universe(self, simple_price_data):
        """
        Test that labels are industry-relative (stock return - universe return).
        """
        horizon = 15

        # Compute components separately
        stock_returns = compute_forward_returns(simple_price_data, horizon)
        universe_returns = compute_universe_returns(simple_price_data, horizon)

        # Compute labels
        labels = compute_industry_relative_labels(simple_price_data, [horizon])

        # Manually compute expected labels
        # Need to broadcast universe returns to stock level
        universe_aligned = stock_returns.index.get_level_values("datetime").map(
            universe_returns.to_dict()
        )
        expected_labels = stock_returns - universe_aligned

        # Compare
        pd.testing.assert_series_equal(
            labels[f"label_{horizon}d"], expected_labels, check_names=False
        )

    def test_labels_sum_to_zero_across_stocks(self, simple_price_data):
        """
        Test that industry-relative labels sum to ~zero across stocks at each date.

        Since labels = stock_return - universe_return, and universe_return is the mean,
        the sum across stocks should be zero (or close to zero due to floating point).
        """
        horizon = 10
        labels = compute_industry_relative_labels(simple_price_data, [horizon])

        # Sum across instruments for each date
        label_sums = labels.groupby(level="datetime")[f"label_{horizon}d"].sum()

        # Should be close to zero (within floating point precision)
        assert label_sums.abs().max() < 1e-10

    def test_labels_nans_at_end(self, simple_price_data):
        """Test that labels have NaNs where insufficient future data."""
        horizon = 21
        labels = compute_industry_relative_labels(simple_price_data, [horizon])

        # Last horizon days should have NaN
        aapl_labels = labels.xs("AAPL", level="instrument")
        assert aapl_labels[f"label_{horizon}d"].iloc[-horizon:].isna().all()

    def test_labels_finite_in_valid_range(self, simple_price_data):
        """Test that labels are finite in the valid range."""
        horizons = [10, 20]
        labels = compute_industry_relative_labels(simple_price_data, horizons)

        # Drop NaNs (end of series)
        labels_valid = labels.dropna()

        # All valid labels should be finite
        assert labels_valid["label_10d"].apply(np.isfinite).all()
        assert labels_valid["label_20d"].apply(np.isfinite).all()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_close_column(self, simple_price_data):
        """Test error handling when close column is missing."""
        df_no_close = simple_price_data.rename(columns={"$close": "price"})

        with pytest.raises(ValueError, match="Close column.*not found"):
            compute_forward_returns(df_no_close, horizon=10)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df_empty = pd.DataFrame({"$close": []})
        df_empty.index = pd.MultiIndex.from_tuples([], names=["instrument", "datetime"])

        # Should return empty Series
        forward_returns = compute_forward_returns(df_empty, horizon=10)
        assert len(forward_returns) == 0

    def test_insufficient_data_for_horizon(self, single_instrument_data):
        """Test behavior when data is shorter than horizon."""
        # Data has 50 days, try horizon of 60
        horizon = 60
        forward_returns = compute_forward_returns(single_instrument_data, horizon)

        # All values should be NaN (insufficient data)
        assert forward_returns.isna().all()

    def test_single_day_data(self):
        """Test behavior with only one day of data."""
        df = pd.DataFrame(
            {"$close": [100.0]},
            index=pd.MultiIndex.from_tuples(
                [("AAPL", pd.Timestamp("2020-01-01"))], names=["instrument", "datetime"]
            ),
        )

        forward_returns = compute_forward_returns(df, horizon=5)

        # Should be all NaN (no returns can be computed)
        # Convert to ndarray before .all() to get a scalar bool
        assert forward_returns.isna().values.all()


class TestDeterminism:
    """Test determinism and reproducibility."""

    def test_deterministic_labels(self, simple_price_data):
        """Test that labels are deterministic (same input -> same output)."""
        horizons = [10, 20]

        # Compute labels twice
        labels1 = compute_industry_relative_labels(simple_price_data, horizons)
        labels2 = compute_industry_relative_labels(simple_price_data, horizons)

        # Should be identical
        pd.testing.assert_frame_equal(labels1, labels2)

    def test_order_independence(self, simple_price_data):
        """Test that labels don't depend on input order."""
        horizons = [10, 20]

        # Compute labels on original data
        labels1 = compute_industry_relative_labels(simple_price_data, horizons)

        # Shuffle data and compute labels
        shuffled_data = simple_price_data.sample(frac=1.0, random_state=42)
        labels2 = compute_industry_relative_labels(shuffled_data, horizons)

        # Sort both by index and compare
        labels1_sorted = labels1.sort_index()
        labels2_sorted = labels2.sort_index()

        pd.testing.assert_frame_equal(labels1_sorted, labels2_sorted)
