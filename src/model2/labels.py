"""
Labels Module for Model 2: Industry-Relative Forward Returns

Implements Chunk 1 of Phase 3 breakdown:
- Compute industry-relative forward returns: y_{i,t}^{(k)} = sum(r_{i,t+1:t+k}) - sum(R_ind_{i,t+1:t+k})
- Horizons: k ∈ {21, 63} days
- Industry proxy: equal-weighted universe returns (Sprint 0 baseline)
- No look-ahead bias: labels use only future returns aligned to prediction date t

Per specs Section 1 (Sprint 0) and theory.md Section 4.1.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import qlib
from qlib.data import D

logger = logging.getLogger(__name__)


def compute_forward_returns(
    df: pd.DataFrame,
    horizon: int,
    close_col: str = "$close",
) -> pd.Series:
    """
    Compute forward cumulative returns over the next k days.

    For each date t, computes: sum(r_{i,t+1:t+k}) where r = log returns.
    The value at date t represents the sum of returns from t+1 through t+k (strictly future).

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        horizon: Number of days to look forward (e.g., 21, 63)
        close_col: Column name for close prices (default: $close per Qlib)

    Returns:
        Series of forward cumulative returns aligned to date t
        Returns NaN for dates where full horizon is not available

    Note:
        - No look-ahead: returns at t+1, ..., t+k only
        - Log returns are used for better numerical properties
        - Groupby instrument to handle panel data correctly
        - Alignment: value at index t = sum(r_{t+1}, ..., r_{t+k})

    Implementation:
        1. Compute log returns: r_t = ln(P_t / P_{t-1})
        2. Rolling sum over window=horizon
        3. Shift entire result by -horizon to align: position t gets sum(r_{t+1:t+k})
    """
    if close_col not in df.columns:
        raise ValueError(f"Close column '{close_col}' not found in DataFrame")

    # Group by instrument to compute returns within each time series
    def compute_instrument_forward_returns(group_df: pd.DataFrame) -> pd.Series:
        """Compute forward returns for a single instrument."""
        # Sort by date to ensure chronological order
        # Handle both MultiIndex and plain DatetimeIndex
        if isinstance(group_df.index, pd.MultiIndex):
            group_df = group_df.sort_index(level="datetime")
        else:
            group_df = group_df.sort_index()

        # Compute log returns: r_t = ln(P_t / P_{t-1})
        log_returns = np.log(group_df[close_col] / group_df[close_col].shift(1))

        # Compute forward cumulative returns: sum(r_{t+1:t+k})
        # Rolling sum on unshifted returns gives sum(r_{t-horizon+1:t}) at position t
        # Shifting by -horizon moves this to position t-horizon
        # So sum(r_{t+1:t+horizon}) ends up at position t
        forward_cumsum = (
            log_returns.rolling(window=horizon, min_periods=horizon).sum().shift(-horizon)
        )

        return forward_cumsum

    # Apply to each instrument group
    if isinstance(df.index, pd.MultiIndex):
        forward_returns = df.groupby(level="instrument", group_keys=False).apply(
            compute_instrument_forward_returns
        )
    else:
        # If not MultiIndex, assume single instrument
        forward_returns = compute_instrument_forward_returns(df)

    return forward_returns


def compute_universe_returns(
    df: pd.DataFrame,
    horizon: int,
    close_col: str = "$close",
) -> pd.Series:
    """
    Compute equal-weighted universe returns as industry proxy.

    For Sprint 0, we use equal-weighted returns across all universe stocks
    as a synthetic industry benchmark (per specs: "industry proxies:
    synthetic equal-weighted returns from universe").

    For each date t, computes: R_univ_t^{(k)} = mean_i(sum(r_{i,t+1:t+k}))

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        horizon: Number of days to look forward
        close_col: Column name for close prices

    Returns:
        Series with index=datetime, containing universe forward returns
        Broadcast to all instruments for subtraction

    Note:
        - Equal-weighted: simple mean across all available stocks at each date
        - Future Sprint 1+: replace with actual industry returns
    """
    # First compute forward returns for all stocks
    forward_returns = compute_forward_returns(df, horizon, close_col)

    # Create DataFrame to properly handle MultiIndex
    if isinstance(forward_returns.index, pd.MultiIndex):
        # Group by date and take mean across instruments
        universe_returns = forward_returns.groupby(level="datetime").mean()
    else:
        # Single instrument case - return the forward returns as-is
        universe_returns = forward_returns

    return universe_returns


def compute_industry_relative_labels(
    df: pd.DataFrame,
    horizons: list[int],
    close_col: str = "$close",
) -> pd.DataFrame:
    """
    Compute industry-relative forward return labels for all horizons.

    Implements: y_{i,t}^{(k)} = sum(r_{i,t+1:t+k}) - sum(R_ind_{i,t+1:t+k})

    Where:
    - r_{i,t} = log return of stock i at time t
    - R_ind_{i,t} = log return of stock i's industry (equal-weighted proxy in S0)
    - k ∈ horizons (typically [21, 63])

    Args:
        df: DataFrame with MultiIndex (instrument, datetime) and close prices
        horizons: List of forward return horizons in days (e.g., [21, 63])
        close_col: Column name for close prices

    Returns:
        DataFrame with columns [label_21d, label_63d, ...]
        Index preserved from input (instrument, datetime)

    Example:
        >>> labels = compute_industry_relative_labels(df, horizons=[21, 63])
        >>> labels.columns
        Index(['label_21d', 'label_63d'], dtype='object')
    """
    logger.info(f"Computing industry-relative labels for horizons: {horizons}")

    result_df = df.copy()[[]]  # Empty DataFrame with same index

    for horizon in horizons:
        logger.debug(f"Processing horizon: {horizon} days")

        # Compute stock-level forward returns
        stock_forward_returns = compute_forward_returns(df, horizon, close_col)

        # Compute universe-level forward returns (industry proxy)
        universe_forward_returns = compute_universe_returns(df, horizon, close_col)

        # Align universe returns to stock-level MultiIndex
        # For each date, all stocks get the same universe return
        if isinstance(stock_forward_returns.index, pd.MultiIndex):
            # Map date -> universe_return for broadcasting
            universe_aligned = stock_forward_returns.index.get_level_values("datetime").map(
                universe_forward_returns.to_dict()
            )
        else:
            universe_aligned = universe_forward_returns.values

        # Compute industry-relative label: stock return - universe return
        label_col = f"label_{horizon}d"
        result_df[label_col] = stock_forward_returns - universe_aligned

        # Log statistics
        valid_count = result_df[label_col].notna().sum()
        total_count = len(result_df)
        logger.info(
            f"Horizon {horizon}d: {valid_count}/{total_count} valid labels "
            f"(mean={result_df[label_col].mean():.6f}, "
            f"std={result_df[label_col].std():.6f})"
        )

    return result_df


def build_labels(
    provider_uri: str,
    region: Literal["US", "CN"],
    start_date: str,
    end_date: str,
    horizons: list[int],
    instruments: list[str] | None = None,
) -> pd.DataFrame:
    """
    Main entry point: build industry-relative forward return labels from Qlib data.

    Loads data from Qlib, computes labels, and returns DataFrame with schema:
    - Index: MultiIndex (instrument, datetime)
    - Columns: [label_21d, label_63d]

    Args:
        provider_uri: Path to Qlib data directory (e.g., "data/qlib/us_data" or "~/.qlib/qlib_data/...")
        region: Trading region ("US" or "CN")
        start_date: Start date for data loading (YYYY-MM-DD)
        end_date: End date for data loading (YYYY-MM-DD)
        horizons: List of forward return horizons [21, 63]
        instruments: Optional list of instruments to load; if None, loads universe

    Returns:
        DataFrame with MultiIndex (instrument, datetime) and label columns

    Raises:
        RuntimeError: If Qlib initialization or data loading fails
        ValueError: If required data is missing or invalid

    Note:
        Data is loaded from start_date to end_date without extension.
        Labels requiring future data beyond end_date will be NaN and dropped.
        For full label coverage through end_date, ensure Qlib data extends
        beyond end_date by max(horizons) days.

    Example:
        >>> labels = build_labels(
        ...     provider_uri="data/qlib/us_data",
        ...     region="US",
        ...     start_date="2020-01-01",
        ...     end_date="2024-12-31",
        ...     horizons=[21, 63],
        ... )
        >>> labels.head()
                                    label_21d  label_63d
        instrument datetime
        AAPL       2020-01-02    0.015234   0.042156
                   2020-01-03   -0.003421   0.038735
        ...
    """
    logger.info(f"Building labels for region={region}, period={start_date} to {end_date}")
    logger.info(f"Provider URI: {provider_uri}")

    # Expand and resolve provider path (handles ~ and relative paths)
    provider_path = Path(provider_uri).expanduser().resolve()
    provider_uri_resolved = str(provider_path)

    # Check if path exists
    if not provider_path.exists():
        raise RuntimeError(f"Qlib data directory not found: {provider_uri_resolved}")

    # Normalize region to lowercase for Qlib (expects "us", "cn")
    region_key = region.lower()

    # Initialize Qlib with resolved path and lowercase region
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
            instruments = None  # Let Qlib D.features load all available

    # Load price data from Qlib
    # Note: We load data from start_date to end_date as specified
    # Labels near end_date will be NaN if insufficient future data exists
    # For full coverage, Qlib data should extend beyond end_date by max(horizons)
    max_horizon = max(horizons)
    logger.info(f"Loading data from {start_date} to {end_date}")
    logger.info(f"Note: Labels require {max_horizon} days of future data")

    try:
        # Load close prices for all instruments
        # D.features returns DataFrame with MultiIndex (instrument, datetime)
        df = D.features(
            instruments=instruments,
            fields=["$close"],
            start_time=start_date,
            end_time=end_date,
            freq="day",
        )

        if df is None or df.empty:
            raise ValueError("No data loaded from Qlib")

        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(
            f"Date range: {df.index.get_level_values('datetime').min()} to "
            f"{df.index.get_level_values('datetime').max()}"
        )
        logger.info(f"Number of instruments: {df.index.get_level_values('instrument').nunique()}")

    except Exception as e:
        raise RuntimeError(f"Failed to load data from Qlib: {e}") from e

    # Compute industry-relative labels
    labels = compute_industry_relative_labels(df, horizons=horizons)

    # Drop rows where all labels are NaN (insufficient future data)
    initial_count = len(labels)
    labels = labels.dropna(how="all")
    dropped_count = initial_count - len(labels)

    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows with insufficient future data")

    logger.info(f"Final labels shape: {labels.shape}")

    # Validation: check output schema
    expected_cols = [f"label_{h}d" for h in horizons]
    if not all(col in labels.columns for col in expected_cols):
        raise RuntimeError(f"Missing expected label columns. Got: {labels.columns.tolist()}")

    # Validation: check for reasonable value range (±1000 bps as per acceptance tests)
    for col in expected_cols:
        outliers = labels[col].abs() > 10.0  # 1000% return (extreme outlier)
        if outliers.any():
            outlier_count = outliers.sum()
            logger.warning(f"Found {outlier_count} extreme outliers in {col} (|label| > 1000%)")

    logger.info("Label computation complete")

    return labels


def build_labels_from_config(config: dict) -> pd.DataFrame:
    """
    Build labels from a configuration dictionary (CLI-friendly wrapper).

    Args:
        config: Configuration dict with keys:
            - provider_uri: str
            - region: "US" or "CN"
            - start_date: str (YYYY-MM-DD)
            - end_date: str (YYYY-MM-DD)
            - labels.horizons: list[int]

    Returns:
        DataFrame with labels
    """
    labels_config = config.get("labels", {})
    horizons = labels_config.get("horizons", [21, 63])

    return build_labels(
        provider_uri=config.get("provider_uri", "data/qlib/us_data"),
        region=config.get("region", "US"),
        start_date=config.get("start_date", "2010-01-01"),
        end_date=config.get("end_date", "2024-12-31"),
        horizons=horizons,
        instruments=None,  # Load from universe
    )
