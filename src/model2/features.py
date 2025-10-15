"""
Features Module for Model 2: Technical Features from Price/Volume Data

Implements Chunk 2 of Phase 3 breakdown:
- Technical features: momentum, MA gaps, RSI, realized vol, drawdown stats
- Microstructure: volume ratios, turnover, Amihud illiquidity
- Preprocessing: winsorize [1%, 99%], then industry z-score at each date
- Missing data handling: forward-fill (limit=5), then dropna

Per specs Section 1 (Sprint 0) and theory.md Section 4.2.
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

logger = logging.getLogger(__name__)


def compute_momentum_features(df: pd.DataFrame, close_col: str = "$close") -> pd.DataFrame:
    """
    Compute momentum features: 12-1 month momentum and 1-month reversal.

    Features:
    - mom_12_1: Cumulative return from t-252 to t-21 (excludes most recent month)
    - mom_1m: Cumulative return from t-21 to t (recent month, often mean-reverting)

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        close_col: Column name for close prices

    Returns:
        DataFrame with momentum feature columns
    """
    logger.debug("Computing momentum features")

    def compute_for_instrument(group_df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum for a single instrument."""
        # Sort by date
        if isinstance(group_df.index, pd.MultiIndex):
            group_df = group_df.sort_index(level="datetime")
        else:
            group_df = group_df.sort_index()

        prices = group_df[close_col]

        # 12-1 month momentum: return from t-252 to t-21
        # Cumulative return = (P_t-21 / P_t-252) - 1 = exp(sum(log_ret)) - 1
        ret_12_1 = prices.shift(21) / prices.shift(252) - 1.0

        # 1-month reversal: return from t-21 to t
        ret_1m = prices / prices.shift(21) - 1.0

        result = pd.DataFrame(index=group_df.index)
        result["mom_12_1"] = ret_12_1
        result["mom_1m"] = ret_1m

        return result

    if isinstance(df.index, pd.MultiIndex):
        features = df.groupby(level="instrument", group_keys=False).apply(compute_for_instrument)
    else:
        features = compute_for_instrument(df)

    return features


def compute_ma_gap_features(df: pd.DataFrame, close_col: str = "$close") -> pd.DataFrame:
    """
    Compute moving average gap features.

    Features:
    - ma_gap_20: (price - MA20) / MA20
    - ma_gap_50: (price - MA50) / MA50
    - ma_gap_200: (price - MA200) / MA200

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        close_col: Column name for close prices

    Returns:
        DataFrame with MA gap feature columns
    """
    logger.debug("Computing MA gap features")

    def compute_for_instrument(group_df: pd.DataFrame) -> pd.DataFrame:
        """Compute MA gaps for a single instrument."""
        # Sort by date
        if isinstance(group_df.index, pd.MultiIndex):
            group_df = group_df.sort_index(level="datetime")
        else:
            group_df = group_df.sort_index()

        prices = group_df[close_col]

        result = pd.DataFrame(index=group_df.index)

        for window in [20, 50, 200]:
            ma = prices.rolling(window=window, min_periods=window).mean()
            result[f"ma_gap_{window}"] = (prices - ma) / ma

        return result

    if isinstance(df.index, pd.MultiIndex):
        features = df.groupby(level="instrument", group_keys=False).apply(compute_for_instrument)
    else:
        features = compute_for_instrument(df)

    return features


def compute_rsi(df: pd.DataFrame, close_col: str = "$close", period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS)
    where RS = average_gain / average_loss over period

    When average losses are zero (all gains), RSI = 100.

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        close_col: Column name for close prices
        period: RSI period (default 14)

    Returns:
        Series with RSI values (0-100 scale)
    """
    logger.debug(f"Computing RSI with period={period}")

    def compute_for_instrument(group_df: pd.DataFrame) -> pd.Series:
        """Compute RSI for a single instrument."""
        # Sort by date
        if isinstance(group_df.index, pd.MultiIndex):
            group_df = group_df.sort_index(level="datetime")
        else:
            group_df = group_df.sort_index()

        prices = group_df[close_col]

        # Compute price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Compute exponential moving averages
        avg_gains = gains.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_losses = losses.ewm(span=period, adjust=False, min_periods=period).mean()

        # Handle edge cases for RSI calculation:
        # 1. Flat price (avg_gains == 0, avg_losses == 0): RS = 1 → RSI = 50
        # 2. Only gains (avg_gains > 0, avg_losses == 0): RS = inf → RSI = 100
        # 3. Only losses (avg_gains == 0, avg_losses > 0): RS = 0 → RSI = 0
        # 4. Normal case: RS = avg_gains / avg_losses

        # Clip to avoid division by zero, compute RS
        avg_losses_safe = avg_losses.clip(lower=1e-12)
        rs = avg_gains / avg_losses_safe

        # Special case: flat price (both gains and losses are zero) → RS = 1
        flat_mask = (avg_gains == 0) & (avg_losses == 0)
        rs = rs.where(~flat_mask, 1.0)

        # Special case: only gains (no losses) → RS = inf
        only_gains_mask = (avg_gains > 0) & (avg_losses == 0)
        rs = rs.where(~only_gains_mask, np.inf)

        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Return as pandas Series with proper index
        return pd.Series(rsi, index=group_df.index, name="rsi_14d")

    if isinstance(df.index, pd.MultiIndex):
        rsi = df.groupby(level="instrument", group_keys=False).apply(compute_for_instrument)
    else:
        rsi = compute_for_instrument(df)

    return rsi


def compute_volatility_features(df: pd.DataFrame, close_col: str = "$close") -> pd.DataFrame:
    """
    Compute realized volatility features.

    Features:
    - vol_21d: Realized volatility over 21 days (annualized)
    - vol_63d: Realized volatility over 63 days (annualized)

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        close_col: Column name for close prices

    Returns:
        DataFrame with volatility feature columns
    """
    logger.debug("Computing volatility features")

    def compute_for_instrument(group_df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility for a single instrument."""
        # Sort by date
        if isinstance(group_df.index, pd.MultiIndex):
            group_df = group_df.sort_index(level="datetime")
        else:
            group_df = group_df.sort_index()

        prices = group_df[close_col]

        # Compute log returns
        log_returns = np.log(prices / prices.shift(1))

        result = pd.DataFrame(index=group_df.index)

        # Realized volatility over different windows (annualized)
        # Vol = std(daily_returns) * sqrt(252)
        for window in [21, 63]:
            vol = log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
            result[f"vol_{window}d"] = vol

        return result

    if isinstance(df.index, pd.MultiIndex):
        features = df.groupby(level="instrument", group_keys=False).apply(compute_for_instrument)
    else:
        features = compute_for_instrument(df)

    return features


def compute_drawdown_features(df: pd.DataFrame, close_col: str = "$close") -> pd.DataFrame:
    """
    Compute drawdown statistics.

    Features:
    - max_dd_252d: Maximum drawdown over past 252 days (true max DD in window)
    - dd_from_peak: Current drawdown from recent peak (63-day window)

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        close_col: Column name for close prices

    Returns:
        DataFrame with drawdown feature columns
    """
    logger.debug("Computing drawdown features")

    def compute_for_instrument(group_df: pd.DataFrame) -> pd.DataFrame:
        """Compute drawdown for a single instrument."""
        # Sort by date
        if isinstance(group_df.index, pd.MultiIndex):
            group_df = group_df.sort_index(level="datetime")
        else:
            group_df = group_df.sort_index()

        prices = group_df[close_col]

        result = pd.DataFrame(index=group_df.index)

        # True maximum drawdown within each 252-day window
        # For each window, compute the worst trough (min drawdown)
        def compute_max_dd(window_prices):
            """Compute maximum drawdown within a price window."""
            if len(window_prices) == 0:
                return np.nan
            running_max = window_prices.cummax()
            drawdowns = (window_prices - running_max) / running_max
            return drawdowns.min()

        result["max_dd_252d"] = prices.rolling(window=252, min_periods=252).apply(
            compute_max_dd, raw=False
        )

        # Current drawdown from 63-day peak
        rolling_max_63 = prices.rolling(window=63, min_periods=63).max()
        result["dd_from_peak"] = (prices - rolling_max_63) / rolling_max_63

        return result

    if isinstance(df.index, pd.MultiIndex):
        features = df.groupby(level="instrument", group_keys=False).apply(compute_for_instrument)
    else:
        features = compute_for_instrument(df)

    return features


def compute_microstructure_features(
    df: pd.DataFrame,
    close_col: str = "$close",
    volume_col: str = "$volume",
) -> pd.DataFrame:
    """
    Compute microstructure features from volume and price data.

    Features:
    - vol_ratio_21d: Volume / MA(Volume, 21)
    - turnover_21d: MA(Volume, 21) (proxy for liquidity)
    - amihud_21d: Amihud illiquidity = |return| / dollar_volume over 21 days

    Args:
        df: DataFrame with MultiIndex (instrument, datetime), close prices, and volume
        close_col: Column name for close prices
        volume_col: Column name for volume

    Returns:
        DataFrame with microstructure feature columns
    """
    logger.debug("Computing microstructure features")

    def compute_for_instrument(group_df: pd.DataFrame) -> pd.DataFrame:
        """Compute microstructure for a single instrument."""
        # Sort by date
        if isinstance(group_df.index, pd.MultiIndex):
            group_df = group_df.sort_index(level="datetime")
        else:
            group_df = group_df.sort_index()

        prices = group_df[close_col]
        volume = group_df[volume_col]

        result = pd.DataFrame(index=group_df.index)

        # Volume ratio: current volume / 21-day MA
        vol_ma_21 = volume.rolling(window=21, min_periods=21).mean()
        result["vol_ratio_21d"] = volume / vol_ma_21

        # Turnover proxy: MA(volume, 21)
        result["turnover_21d"] = vol_ma_21

        # Amihud illiquidity: mean(|return| / dollar_volume) over 21 days
        # Dollar volume = price * volume
        log_returns = np.log(prices / prices.shift(1))
        dollar_volume = prices * volume

        # Avoid division by zero
        illiq_daily = log_returns.abs() / dollar_volume.replace(0, np.nan)
        result["amihud_21d"] = illiq_daily.rolling(window=21, min_periods=21).mean()

        return result

    if isinstance(df.index, pd.MultiIndex):
        features = df.groupby(level="instrument", group_keys=False).apply(compute_for_instrument)
    else:
        features = compute_for_instrument(df)

    return features


def winsorize_cross_sectional(
    df: pd.DataFrame,
    col: str,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """
    Winsorize a feature cross-sectionally within each date.

    This avoids look-ahead bias by computing quantiles separately for each date
    across all instruments at that date.

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and feature column
        col: Column name to winsorize
        lower: Lower quantile (default 1%)
        upper: Upper quantile (default 99%)

    Returns:
        Winsorized series
    """

    def winsorize_group(group_series: pd.Series) -> pd.Series:
        """Winsorize a single date group."""
        lower_bound = group_series.quantile(lower)
        upper_bound = group_series.quantile(upper)
        return group_series.clip(lower=lower_bound, upper=upper_bound)

    if isinstance(df.index, pd.MultiIndex):
        # Group by date and winsorize cross-sectionally
        return df.groupby(level="datetime")[col].transform(winsorize_group)
    else:
        # Single date case - compute quantiles on the full series
        return winsorize_group(df[col])


def industry_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Apply industry z-score normalization at each date.

    For Sprint 0, we use universe-wide z-score (no industry grouping).
    Future sprints will use actual industry assignments.

    Formula: z_{i,t} = (x_{i,t} - mean_t(x)) / std_t(x)

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and feature columns
        feature_cols: List of feature column names to normalize

    Returns:
        DataFrame with normalized features
    """
    logger.debug(f"Applying z-score normalization to {len(feature_cols)} features")

    result = df.copy()

    # For each date, compute cross-sectional z-score
    for col in feature_cols:
        if col not in result.columns:
            logger.warning(f"Feature column {col} not found, skipping")
            continue

        # Group by date and normalize
        def zscore_group(group_series: pd.Series) -> pd.Series:
            """Compute z-score within a date group."""
            mean = group_series.mean()
            std = group_series.std()
            # Avoid division by zero
            if std == 0 or pd.isna(std):
                return pd.Series(0.0, index=group_series.index)
            return (group_series - mean) / std

        if isinstance(result.index, pd.MultiIndex):
            result[col] = result.groupby(level="datetime")[col].transform(zscore_group)
        else:
            # Single date case
            result[col] = zscore_group(result[col])

    return result


def handle_missing_data(
    df: pd.DataFrame,
    forward_fill_limit: int = 5,
) -> pd.DataFrame:
    """
    Handle missing data: forward-fill (limit=5), then drop remaining NaNs.

    Args:
        df: DataFrame with potential missing values
        forward_fill_limit: Maximum number of consecutive NaNs to fill

    Returns:
        DataFrame with missing data handled
    """
    logger.debug(f"Handling missing data with forward-fill limit={forward_fill_limit}")

    initial_count = len(df)
    initial_nan_count = df.isna().sum().sum()

    # Forward-fill within each instrument
    if isinstance(df.index, pd.MultiIndex):
        df = df.groupby(level="instrument", group_keys=False).apply(
            lambda x: x.ffill(limit=forward_fill_limit)
        )
    else:
        df = df.ffill(limit=forward_fill_limit)

    after_ffill_nan_count = df.isna().sum().sum()

    # Drop rows with any remaining NaNs
    df = df.dropna()

    final_count = len(df)
    dropped_count = initial_count - final_count

    logger.info(
        f"Missing data handling: "
        f"initial_nans={initial_nan_count}, "
        f"after_ffill={after_ffill_nan_count}, "
        f"rows_dropped={dropped_count}"
    )

    return df


def build_features(
    df: pd.DataFrame,
    winsor_pct: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """
    Build all technical features from price/volume data.

    Feature categories:
    1. Momentum: mom_12_1, mom_1m
    2. MA gaps: ma_gap_20, ma_gap_50, ma_gap_200
    3. RSI: rsi_14d
    4. Volatility: vol_21d, vol_63d
    5. Drawdown: max_dd_252d, dd_from_peak
    6. Microstructure: vol_ratio_21d, turnover_21d, amihud_21d

    Preprocessing:
    1. Compute raw features
    2. Winsorize at specified percentiles (cross-sectionally by date)
    3. Industry z-score normalization
    4. Handle missing data (forward-fill limit=5, then dropna)

    Args:
        df: DataFrame with MultiIndex (instrument, datetime), close prices, and volume
        winsor_pct: Tuple of (lower, upper) percentiles for winsorization

    Returns:
        DataFrame with all features, fully preprocessed
    """
    logger.info("Building technical features")

    # Check required columns
    required_cols = ["$close", "$volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize result with input index
    result = pd.DataFrame(index=df.index)

    # 1. Momentum features
    momentum = compute_momentum_features(df)
    result = result.join(momentum)

    # 2. MA gap features
    ma_gaps = compute_ma_gap_features(df)
    result = result.join(ma_gaps)

    # 3. RSI
    result["rsi_14d"] = compute_rsi(df, period=14)

    # 4. Volatility features
    volatility = compute_volatility_features(df)
    result = result.join(volatility)

    # 5. Drawdown features
    drawdown = compute_drawdown_features(df)
    result = result.join(drawdown)

    # 6. Microstructure features
    microstructure = compute_microstructure_features(df)
    result = result.join(microstructure)

    logger.info(f"Raw features computed: {result.shape[1]} features")

    # Get list of all feature columns
    feature_cols = result.columns.tolist()

    # Preprocessing step 1: Winsorize (cross-sectionally by date)
    logger.info(f"Winsorizing features at {winsor_pct} (cross-sectionally by date)")
    for col in feature_cols:
        result[col] = winsorize_cross_sectional(
            result, col, lower=winsor_pct[0], upper=winsor_pct[1]
        )

    # Preprocessing step 2: Industry z-score (universe-wide in Sprint 0)
    result = industry_zscore(result, feature_cols)

    # Preprocessing step 3: Handle missing data
    result = handle_missing_data(result, forward_fill_limit=5)

    logger.info(f"Final features shape: {result.shape}")
    logger.info(f"Feature columns: {result.columns.tolist()}")

    return result


def build_features_from_qlib(
    provider_uri: str,
    region: Literal["US", "CN"],
    start_date: str,
    end_date: str,
    winsor_pct: tuple[float, float] = (0.01, 0.99),
    instruments: list[str] | None = None,
    lookback_buffer_days: int = 700,
) -> pd.DataFrame:
    """
    Main entry point: build technical features from Qlib data.

    Loads data from Qlib, computes features, and returns DataFrame with schema:
    - Index: MultiIndex (instrument, datetime)
    - Columns: [mom_12_1, mom_1m, ma_gap_20, ..., amihud_21d]

    Args:
        provider_uri: Path to Qlib data directory
        region: Trading region ("US" or "CN")
        start_date: Start date for data loading (YYYY-MM-DD)
        end_date: End date for data loading (YYYY-MM-DD)
        winsor_pct: Tuple of (lower, upper) percentiles for winsorization
        instruments: Optional list of instruments to load
        lookback_buffer_days: Calendar days to extend start_date for feature computation (default 700)

    Returns:
        DataFrame with MultiIndex (instrument, datetime) and feature columns
        Trimmed to the requested [start_date, end_date] range

    Raises:
        RuntimeError: If Qlib initialization or data loading fails
        ValueError: If required data is missing or invalid

    Note:
        Data is automatically loaded from an extended start date to ensure sufficient
        lookback for feature computation (252-day momentum + 200-day MA = ~500 trading days).
        Final features are trimmed back to the requested [start_date, end_date] range.

    Example:
        >>> features = build_features_from_qlib(
        ...     provider_uri="data/qlib/us_data",
        ...     region="US",
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     winsor_pct=(0.01, 0.99),
        ... )
        >>> features.head()
                                    mom_12_1  mom_1m  ma_gap_20  ...
        instrument datetime
        AAPL       2020-01-02    0.25       -0.03   0.05       ...
        ...
    """
    logger.info(f"Building features for region={region}, period={start_date} to {end_date}")
    logger.info(f"Provider URI: {provider_uri}")

    # Expand and resolve provider path
    provider_path = Path(provider_uri).expanduser().resolve()
    provider_uri_resolved = str(provider_path)

    # Check if path exists
    if not provider_path.exists():
        raise RuntimeError(f"Qlib data directory not found: {provider_uri_resolved}")

    # Normalize region to lowercase for Qlib
    region_key = region.lower()

    # Initialize Qlib
    try:
        qlib.init(provider_uri=provider_uri_resolved, region=region_key)
        logger.info("Qlib initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Qlib: {e}") from e

    # Load instruments from universe if not specified
    if instruments is None:
        universe_file = Path(f"data/universe_{region_key}.parquet")
        if universe_file.exists():
            logger.info(f"Loading universe from {universe_file}")
            universe_df = pd.read_parquet(universe_file)
            instruments = sorted(universe_df["instrument"].unique().tolist())
            logger.info(f"Loaded {len(instruments)} instruments from universe")
        else:
            logger.warning(f"Universe file not found: {universe_file}, using all instruments")
            instruments = None

    # Calculate extended start date for feature lookback
    # Maximum lookback: 252 days (12-month momentum) + 200 days (MA200) = 452 trading days
    # Use 700 calendar days buffer to ensure ~500 trading days (accounting for weekends/holidays)
    start_dt = pd.to_datetime(start_date)
    extended_start_date = (start_dt - timedelta(days=lookback_buffer_days)).strftime("%Y-%m-%d")

    logger.info(f"Extended start date for lookback: {extended_start_date}")
    logger.info(f"Requested date range: {start_date} to {end_date}")
    logger.info(
        "Note: Data will be loaded from extended start date then trimmed to requested range"
    )

    # Load price and volume data from Qlib with extended window
    try:
        df = D.features(
            instruments=instruments,
            fields=["$close", "$volume"],
            start_time=extended_start_date,
            end_time=end_date,
            freq="day",
        )

        if df is None or df.empty:
            raise ValueError("No data loaded from Qlib")

        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(
            f"Loaded date range: {df.index.get_level_values('datetime').min()} to "
            f"{df.index.get_level_values('datetime').max()}"
        )
        logger.info(f"Number of instruments: {df.index.get_level_values('instrument').nunique()}")

    except Exception as e:
        raise RuntimeError(f"Failed to load data from Qlib: {e}") from e

    # Build features on full data range (including lookback)
    features = build_features(df, winsor_pct=winsor_pct)

    # Trim features to requested date range
    if isinstance(features.index, pd.MultiIndex):
        # Filter by datetime level
        date_mask = (features.index.get_level_values("datetime") >= start_date) & (
            features.index.get_level_values("datetime") <= end_date
        )
        features = features.loc[date_mask]
    else:
        # Single instrument case - filter by index
        features = features.loc[(features.index >= start_date) & (features.index <= end_date)]

    logger.info(f"Features trimmed to requested range: {features.shape}")
    logger.info(
        f"Final date range: {features.index.get_level_values('datetime').min()} to "
        f"{features.index.get_level_values('datetime').max()}"
    )
    logger.info("Feature computation complete")

    return features


def build_features_from_config(config: dict) -> pd.DataFrame:
    """
    Build features from a configuration dictionary (CLI-friendly wrapper).

    Args:
        config: Configuration dict with keys:
            - provider_uri: str
            - region: "US" or "CN"
            - start_date: str (YYYY-MM-DD)
            - end_date: str (YYYY-MM-DD)
            - features.winsor_pct: list[float, float]
            - features.lookback_buffer_days: int (optional, default 700)

    Returns:
        DataFrame with features
    """
    features_config = config.get("features", {})
    winsor_pct = tuple(features_config.get("winsor_pct", [0.01, 0.99]))
    lookback_buffer_days = features_config.get("lookback_buffer_days", 700)

    return build_features_from_qlib(
        provider_uri=config.get("provider_uri", "data/qlib/us_data"),
        region=config.get("region", "US"),
        start_date=config.get("start_date", "2010-01-01"),
        end_date=config.get("end_date", "2024-12-31"),
        winsor_pct=winsor_pct,
        lookback_buffer_days=lookback_buffer_days,
        instruments=None,
    )
