"""
Unit tests for Model 2 features module.

Tests:
- Technical features: momentum, MA gaps, RSI, volatility, drawdown
- Microstructure features: volume ratios, turnover, Amihud
- Preprocessing: winsorization, z-score normalization
- Missing data handling
- Schema validation
- Edge cases

Per Phase 3 Chunk 2 acceptance criteria.
"""

import numpy as np
import pandas as pd
import pytest

from src.model2.features import (
    build_features,
    compute_drawdown_features,
    compute_ma_gap_features,
    compute_microstructure_features,
    compute_momentum_features,
    compute_rsi,
    compute_volatility_features,
    handle_missing_data,
    industry_zscore,
    winsorize_cross_sectional,
)


@pytest.fixture
def simple_price_volume_data():
    """
    Create simple price and volume data for testing.

    Two instruments over 300 days with known patterns.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    instruments = ["AAPL", "GOOGL"]

    # Create MultiIndex
    index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])

    # Create price series with trend
    prices = []
    volumes = []
    for inst in instruments:
        base_price = 100.0 if inst == "AAPL" else 200.0
        base_volume = 1_000_000

        # Create prices with trend and noise
        inst_prices = [
            base_price * (1 + 0.001 * i + np.random.normal(0, 0.01)) for i in range(len(dates))
        ]

        # Create volumes
        inst_volumes = [base_volume * (1 + np.random.normal(0, 0.1)) for _ in range(len(dates))]

        prices.extend(inst_prices)
        volumes.extend(inst_volumes)

    df = pd.DataFrame({"$close": prices, "$volume": volumes}, index=index)
    return df


@pytest.fixture
def single_instrument_data():
    """Create data for a single instrument."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")

    prices = [100 * (1 + 0.001 * i + np.random.normal(0, 0.01)) for i in range(len(dates))]
    volumes = [1_000_000 * (1 + np.random.normal(0, 0.1)) for _ in range(len(dates))]

    df = pd.DataFrame({"$close": prices, "$volume": volumes}, index=dates)
    df.index.name = "datetime"

    return df


class TestComputeMomentumFeatures:
    """Test momentum features calculation."""

    def test_momentum_shape(self, simple_price_volume_data):
        """Test that momentum features have correct shape."""
        momentum = compute_momentum_features(simple_price_volume_data)

        assert isinstance(momentum, pd.DataFrame)
        assert len(momentum) == len(simple_price_volume_data)
        assert "mom_12_1" in momentum.columns
        assert "mom_1m" in momentum.columns

    def test_momentum_12_1_excludes_recent_month(self):
        """Test that 12-1 momentum excludes the most recent month."""
        # Create simple data: constant price growth
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        prices = [100 * (1.001**i) for i in range(300)]

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        momentum = compute_momentum_features(df)

        # 12-1 momentum should use prices from t-252 to t-21
        # At index 252: should use prices at index 231 and index 0
        # Return = (P_231 / P_0) - 1
        expected_return = (prices[231] / prices[0]) - 1.0
        actual_return = momentum["mom_12_1"].iloc[252]

        assert actual_return == pytest.approx(expected_return, rel=1e-6)

    def test_momentum_nans_for_insufficient_data(self, simple_price_volume_data):
        """Test that momentum has NaNs when insufficient lookback data."""
        momentum = compute_momentum_features(simple_price_volume_data)

        # First 252 rows should have NaN for mom_12_1 (no 252-day lookback)
        aapl_momentum = momentum.xs("AAPL", level="instrument")
        assert aapl_momentum["mom_12_1"].iloc[:252].isna().all()

        # First 21 rows should have NaN for mom_1m
        assert aapl_momentum["mom_1m"].iloc[:21].isna().all()


class TestComputeMAGapFeatures:
    """Test moving average gap features."""

    def test_ma_gap_shape(self, simple_price_volume_data):
        """Test that MA gap features have correct shape."""
        ma_gaps = compute_ma_gap_features(simple_price_volume_data)

        assert isinstance(ma_gaps, pd.DataFrame)
        assert len(ma_gaps) == len(simple_price_volume_data)
        assert "ma_gap_20" in ma_gaps.columns
        assert "ma_gap_50" in ma_gaps.columns
        assert "ma_gap_200" in ma_gaps.columns

    def test_ma_gap_calculation(self):
        """Test MA gap calculation with known values."""
        # Create flat prices at 100
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = [100.0] * 50

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        ma_gaps = compute_ma_gap_features(df)

        # For flat prices, all MA gaps should be 0 (after warmup)
        assert ma_gaps["ma_gap_20"].iloc[20:].abs().max() < 1e-10


class TestComputeRSI:
    """Test RSI calculation."""

    def test_rsi_shape(self, simple_price_volume_data):
        """Test that RSI has correct shape."""
        rsi = compute_rsi(simple_price_volume_data)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(simple_price_volume_data)

    def test_rsi_range(self, simple_price_volume_data):
        """Test that RSI is in valid range [0, 100]."""
        rsi = compute_rsi(simple_price_volume_data)

        # Drop NaNs
        rsi_valid = rsi.dropna()

        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_rsi_all_gains(self):
        """Test RSI when all price changes are gains (RSI = 100)."""
        # Create constantly increasing prices
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        prices = [100 * (1.01**i) for i in range(50)]

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        rsi = compute_rsi(df, period=14)

        # After warmup, RSI should be close to 100
        assert rsi.iloc[20:].min() > 95.0

    def test_rsi_zero_loss_handling(self):
        """Test that RSI handles zero average losses correctly."""
        # Create prices with only gains
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = list(range(100, 130))  # Strictly increasing

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        rsi = compute_rsi(df, period=14)

        # Should not have NaN after warmup (zero-loss should -> RSI=100)
        assert not rsi.iloc[15:].isna().any()

    def test_rsi_flat_prices(self):
        """Test that RSI returns 50 for flat prices (zero gains and zero losses)."""
        # Create constant prices (no movement)
        dates = pd.date_range("2020-01-01", periods=30, freq="D")
        prices = [100.0] * 30  # Constant price

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        rsi = compute_rsi(df, period=14)

        # After warmup period, RSI should be 50 (neutral)
        # RS = 1 when avg_gains == avg_losses == 0
        # RSI = 100 - 100/(1 + 1) = 50
        rsi_after_warmup = rsi.iloc[15:]
        assert np.allclose(rsi_after_warmup.to_numpy(), 50.0, atol=1e-10)


class TestComputeVolatilityFeatures:
    """Test volatility features."""

    def test_volatility_shape(self, simple_price_volume_data):
        """Test that volatility features have correct shape."""
        volatility = compute_volatility_features(simple_price_volume_data)

        assert isinstance(volatility, pd.DataFrame)
        assert len(volatility) == len(simple_price_volume_data)
        assert "vol_21d" in volatility.columns
        assert "vol_63d" in volatility.columns

    def test_volatility_positive(self, simple_price_volume_data):
        """Test that volatility is always positive."""
        volatility = compute_volatility_features(simple_price_volume_data)

        # Drop NaNs
        volatility_valid = volatility.dropna()

        assert (volatility_valid["vol_21d"] >= 0).all()
        assert (volatility_valid["vol_63d"] >= 0).all()

    def test_volatility_constant_prices(self):
        """Test volatility for constant prices (vol = 0)."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = [100.0] * 100

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        volatility = compute_volatility_features(df)

        # Volatility should be 0 (or very small due to numerical precision)
        assert volatility["vol_21d"].iloc[21:].abs().max() < 1e-10


class TestComputeDrawdownFeatures:
    """Test drawdown features."""

    def test_drawdown_shape(self, simple_price_volume_data):
        """Test that drawdown features have correct shape."""
        drawdown = compute_drawdown_features(simple_price_volume_data)

        assert isinstance(drawdown, pd.DataFrame)
        assert len(drawdown) == len(simple_price_volume_data)
        assert "max_dd_252d" in drawdown.columns
        assert "dd_from_peak" in drawdown.columns

    def test_drawdown_negative_or_zero(self, simple_price_volume_data):
        """Test that drawdown is always negative or zero."""
        drawdown = compute_drawdown_features(simple_price_volume_data)

        # Drop NaNs
        drawdown_valid = drawdown.dropna()

        assert (drawdown_valid["max_dd_252d"] <= 0).all()
        assert (drawdown_valid["dd_from_peak"] <= 0).all()

    def test_drawdown_at_peak_is_zero(self):
        """Test that drawdown is zero at new peaks."""
        # Create constantly increasing prices (always at new peak)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = list(range(100, 200))

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        drawdown = compute_drawdown_features(df)

        # dd_from_peak should be 0 for constantly increasing prices
        assert drawdown["dd_from_peak"].iloc[63:].abs().max() < 1e-10

    def test_max_drawdown_captures_worst_trough(self):
        """Test that max_dd captures the worst trough in the window."""
        # Create prices with a known drawdown pattern
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        # Start at 100, go to 150, drop to 75 (50% DD from peak), then recover to 125
        prices = (
            [100 + i for i in range(50)]  # Rise to 150
            + [150 - i for i in range(75)]  # Drop to 75
            + [75 + i for i in range(50)]  # Rise to 125
            + [125] * 125  # Flat
        )

        df = pd.DataFrame({"$close": prices}, index=dates)
        df.index.name = "datetime"

        drawdown = compute_drawdown_features(df)

        # At the trough (index 125), max_dd_252d should capture the 50% drop
        # Peak was at 150 (index 49), trough at 75 (index 124)
        # DD = (75 - 150) / 150 = -0.5
        max_dd_at_trough = drawdown["max_dd_252d"].iloc[252]

        assert max_dd_at_trough < -0.45  # Should be close to -0.5


class TestComputeMicrostructureFeatures:
    """Test microstructure features."""

    def test_microstructure_shape(self, simple_price_volume_data):
        """Test that microstructure features have correct shape."""
        microstructure = compute_microstructure_features(simple_price_volume_data)

        assert isinstance(microstructure, pd.DataFrame)
        assert len(microstructure) == len(simple_price_volume_data)
        assert "vol_ratio_21d" in microstructure.columns
        assert "turnover_21d" in microstructure.columns
        assert "amihud_21d" in microstructure.columns

    def test_microstructure_positive_values(self, simple_price_volume_data):
        """Test that microstructure features are positive."""
        microstructure = compute_microstructure_features(simple_price_volume_data)

        # Drop NaNs
        microstructure_valid = microstructure.dropna()

        # Volume ratio and turnover should be positive
        assert (microstructure_valid["vol_ratio_21d"] > 0).all()
        assert (microstructure_valid["turnover_21d"] > 0).all()
        # Amihud illiquidity should be non-negative
        assert (microstructure_valid["amihud_21d"] >= 0).all()


class TestWinsorization:
    """Test winsorization preprocessing."""

    def test_winsorize_cross_sectional_clips_extremes(self):
        """Test that winsorization clips extreme values."""
        # Create data with outliers
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        instruments = [f"STOCK{i}" for i in range(100)]

        index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])

        # Create feature with outliers: 98 normal values + 1 very high + 1 very low
        np.random.seed(42)
        values = []
        for _ in dates:
            date_values = list(np.random.normal(0, 1, 98)) + [10.0, -10.0]
            values.extend(date_values)

        df = pd.DataFrame({"feature": values}, index=index)

        # Winsorize at 1% and 99%
        winsorized = winsorize_cross_sectional(df, "feature", lower=0.01, upper=0.99)

        # Extreme values should be clipped (winsorized is a Series)
        max_val = winsorized.max()
        min_val = winsorized.min()
        assert max_val < 10.0
        assert min_val > -10.0

    def test_winsorize_no_lookahead(self):
        """Test that winsorization is cross-sectional (no look-ahead)."""
        # Create data where each date has different distribution
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        instruments = [f"STOCK{i}" for i in range(10)]

        index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])

        # Date 1: values 0-9, Date 2: values 10-19
        values = list(range(10)) + list(range(10, 20))

        df = pd.DataFrame({"feature": values}, index=index)

        winsorized = winsorize_cross_sectional(df, "feature", lower=0.1, upper=0.9)

        # Date 1 and Date 2 should be winsorized separately
        date1_values = winsorized.xs(dates[0], level="datetime")
        date2_values = winsorized.xs(dates[1], level="datetime")

        # Check that max values are different (clipped separately per date)
        assert date1_values.max() < date2_values.max()


class TestIndustryZScore:
    """Test industry z-score normalization."""

    def test_zscore_zero_mean_unit_std(self):
        """Test that z-score normalization produces zero mean and unit std."""
        # Create data
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        instruments = [f"STOCK{i}" for i in range(50)]

        index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])

        np.random.seed(42)
        values = np.random.normal(5, 2, len(index))

        df = pd.DataFrame({"feature": values}, index=index)

        # Apply z-score
        normalized = industry_zscore(df, ["feature"])

        # For each date, mean should be ~0 and std should be ~1
        for date in dates:
            date_values = normalized.xs(date, level="datetime")["feature"]
            assert date_values.mean() == pytest.approx(0.0, abs=1e-10)
            assert date_values.std() == pytest.approx(1.0, abs=1e-10)

    def test_zscore_no_lookahead(self):
        """Test that z-score is computed cross-sectionally (no look-ahead)."""
        # Create data where each date has different mean
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        instruments = [f"STOCK{i}" for i in range(10)]

        index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])

        # Date 1: mean=0, Date 2: mean=100
        values = list(range(-5, 5)) + list(range(95, 105))

        df = pd.DataFrame({"feature": values}, index=index)

        normalized = industry_zscore(df, ["feature"])

        # Both dates should have mean ~0 after normalization
        date1_mean = normalized.xs(dates[0], level="datetime")["feature"].mean()
        date2_mean = normalized.xs(dates[1], level="datetime")["feature"].mean()

        assert date1_mean == pytest.approx(0.0, abs=1e-10)
        assert date2_mean == pytest.approx(0.0, abs=1e-10)


class TestHandleMissingData:
    """Test missing data handling."""

    def test_forward_fill_fills_gaps(self):
        """Test that forward fill fills small gaps."""
        # Create data with missing values
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        values = [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]

        df = pd.DataFrame({"feature": values}, index=dates)
        df.index.name = "datetime"

        # Forward fill with limit=5
        filled = handle_missing_data(df, forward_fill_limit=5)

        # Gaps of 2 should be filled
        assert len(filled) > 0
        assert not filled["feature"].iloc[0:5].isna().any()

    def test_forward_fill_drops_long_gaps(self):
        """Test that forward fill drops rows with long gaps."""
        # Create data with a long gap
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        values = [1.0] * 5 + [np.nan] * 10 + [2.0] * 5

        df = pd.DataFrame({"feature": values}, index=dates)
        df.index.name = "datetime"

        # Forward fill with limit=5 (gap of 10 exceeds limit)
        filled = handle_missing_data(df, forward_fill_limit=5)

        # Long gap should result in dropped rows
        assert len(filled) < len(df)


class TestBuildFeatures:
    """Test the full feature building pipeline."""

    def test_build_features_schema(self, simple_price_volume_data):
        """Test that build_features produces correct schema."""
        features = build_features(simple_price_volume_data)

        # Check expected columns exist
        expected_features = [
            "mom_12_1",
            "mom_1m",
            "ma_gap_20",
            "ma_gap_50",
            "ma_gap_200",
            "rsi_14d",
            "vol_21d",
            "vol_63d",
            "max_dd_252d",
            "dd_from_peak",
            "vol_ratio_21d",
            "turnover_21d",
            "amihud_21d",
        ]

        for feature in expected_features:
            assert feature in features.columns

    def test_build_features_no_nans(self, simple_price_volume_data):
        """Test that build_features drops all NaNs."""
        features = build_features(simple_price_volume_data)

        # After preprocessing, there should be no NaNs
        assert not features.isna().any().any()

    def test_build_features_normalized(self, simple_price_volume_data):
        """Test that features are normalized (z-scored) where variance exists."""
        features = build_features(simple_price_volume_data)

        # For each date and feature, check that cross-sectional mean is ~0
        # and std is ~1 (within tolerance) for dates with actual variance
        dates = features.index.get_level_values("datetime").unique()

        # Check a subset of dates that should have valid data (skip very early warm-up)
        # With 300 days of data and max look back of 252, we have ~48 valid dates
        # Check the last 10 dates which should definitely be post-warm-up
        for date in dates[-10:]:
            date_features = features.xs(date, level="datetime")

            for col in date_features.columns:
                std = date_features[col].std()

                # Only check normalization if there's actual variance (std > 0)
                if std > 0:
                    mean = date_features[col].mean()
                    # Mean should be close to 0
                    assert abs(mean) < 0.1, f"Date {date}, col {col}: mean={mean}"
                    # Std should be close to 1
                    assert abs(std - 1.0) < 0.2, f"Date {date}, col {col}: std={std}"

    def test_build_features_single_instrument(self, single_instrument_data):
        """Test build_features with single instrument."""
        features = build_features(single_instrument_data)

        # Should work without errors
        assert len(features) > 0
        assert "mom_12_1" in features.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_volume_column(self, simple_price_volume_data):
        """Test error handling when volume column is missing."""
        df_no_volume = simple_price_volume_data.drop(columns=["$volume"])

        with pytest.raises(ValueError, match="Missing required columns"):
            build_features(df_no_volume)

    def test_insufficient_data_for_features(self):
        """Test behavior when data is too short for features."""
        # Only 50 days of data (not enough for 252-day momentum)
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        # Use varying prices (not constant) to properly test insufficient data
        prices = [100.0 + i * 0.1 for i in range(50)]
        volumes = [1_000_000.0] * 50

        df = pd.DataFrame({"$close": prices, "$volume": volumes}, index=dates)
        df.index.name = "datetime"

        features = build_features(df)

        # Should have features computed, but some rows dropped due to NaNs
        # (e.g., mom_12_1 will be all NaN)
        assert len(features) < len(df)

    def test_constant_prices(self):
        """Test behavior with constant prices."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        prices = [100.0] * 300
        volumes = [1_000_000.0] * 300

        df = pd.DataFrame({"$close": prices, "$volume": volumes}, index=dates)
        df.index.name = "datetime"

        # Should not crash
        features = build_features(df)

        # Momentum, MA gaps, drawdown should be 0
        assert len(features) > 0


class TestDeterminism:
    """Test determinism and reproducibility."""

    def test_deterministic_features(self, simple_price_volume_data):
        """Test that features are deterministic."""
        features1 = build_features(simple_price_volume_data)
        features2 = build_features(simple_price_volume_data)

        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)

    def test_order_independence(self, simple_price_volume_data):
        """Test that features don't depend on input order."""
        # Compute features on original data
        features1 = build_features(simple_price_volume_data)

        # Shuffle data and compute features
        shuffled_data = simple_price_volume_data.sample(frac=1.0, random_state=42)
        features2 = build_features(shuffled_data)

        # Sort both by index and compare
        features1_sorted = features1.sort_index()
        features2_sorted = features2.sort_index()

        pd.testing.assert_frame_equal(features1_sorted, features2_sorted)
