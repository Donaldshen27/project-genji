"""
Universe construction with monthly reconstitution.

Implements:
- Monthly reconstitution (last trading day of each month)
- Eligibility filters (price, volume, ST stocks)
- Top N=800 selection by market cap
- Industry assignment (frozen-at-first-seen for Sprint 0)

CLI: python -m src.data.build_universe --config configs/data_{region}.yaml --output data/universe_{region}.parquet
"""

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import yaml
from tqdm import tqdm

from src.utils.logging import get_logger, setup_logging


def normalize_config(raw_config: dict) -> dict:
    """
    Convert datetime.date objects to ISO strings for config hashing.
    """

    def normalize_value(value):
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        elif isinstance(value, dict):
            return {k: normalize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [normalize_value(item) for item in value]
        return value

    return {key: normalize_value(value) for key, value in raw_config.items()}


def get_month_end_dates(start_date: str, end_date: str, calendar_name: str) -> pd.DatetimeIndex:
    """
    Get last trading day of each month in the date range.

    Args:
        start_date: Start date as ISO string
        end_date: End date as ISO string
        calendar_name: Trading calendar (NYSE, SSE, etc.)

    Returns:
        DatetimeIndex of month-end trading days (timezone-naive)
    """
    logger = get_logger(__name__)

    # Get trading calendar
    if calendar_name == "NYSE":
        calendar = mcal.get_calendar("NYSE")
    elif calendar_name == "SSE":
        calendar = mcal.get_calendar("SSE")
    else:
        logger.warning(f"Unknown calendar {calendar_name}, defaulting to NYSE")
        calendar = mcal.get_calendar("NYSE")

    # Get all trading days in range
    trading_days = calendar.valid_days(start_date=start_date, end_date=end_date)

    # Group by month and get last trading day of each month
    month_ends = trading_days.to_series().groupby(pd.Grouper(freq="M")).last()
    month_ends = pd.DatetimeIndex(month_ends)

    # Strip timezone to match parquet data (which is timezone-naive)
    month_ends = month_ends.tz_localize(None).normalize()

    logger.info(f"Found {len(month_ends)} month-end dates from {start_date} to {end_date}")
    return month_ends


def load_ticker_data(ticker: str, instruments_dir: Path) -> pd.DataFrame | None:
    """
    Load ticker data from parquet file.

    Returns:
        DataFrame with date index (timezone-naive) and columns: open, high, low, close, volume, adj_close, adj_factor
        or None if file doesn't exist or is invalid
    """
    logger = get_logger(__name__)

    ticker_file = instruments_dir / f"{ticker}.parquet"
    if not ticker_file.exists():
        logger.debug(f"{ticker}: File not found at {ticker_file}")
        return None

    try:
        df = pd.read_parquet(ticker_file)
        if df.empty:
            return None
        # Ensure date index
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        # Ensure timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        logger.error(f"{ticker}: Failed to load data: {e}")
        return None


def compute_adv_usd(df: pd.DataFrame, window: int = 20, is_cn: bool = False) -> pd.Series:
    """
    Compute average daily volume in USD.

    Args:
        df: DataFrame with close and volume columns
        window: Rolling window for average (default 20 days)
        is_cn: If True, convert CNY to USD (use amount field if available)

    Returns:
        Series of ADV in USD
    """
    if is_cn and "amount" in df.columns:
        # For CN: amount is in CNY, convert to USD (approx 6.5 CNY/USD)
        # ADV in USD = rolling_mean(amount in CNY / 6.5)
        adv_cny = df["amount"].rolling(window=window, min_periods=window).mean()
        return adv_cny / 6.5
    else:
        # For US: volume * close = dollar volume
        dollar_volume = df["close"] * df["volume"]
        return dollar_volume.rolling(window=window, min_periods=window).mean()


def get_industry_classification(ticker: str, region: str) -> str:
    """
    Get industry classification for ticker.

    For Sprint 0: Returns simple placeholder industry.
    Production should use:
    - US: yfinance sector or GICS
    - CN: Shenwan Level 1 classification

    Args:
        ticker: Stock ticker/code
        region: US or CN

    Returns:
        Industry ID string
    """
    logger = get_logger(__name__)

    # Sprint 0 placeholder: Use first digit of ticker as industry proxy
    # Production should fetch from yfinance (US) or tushare (CN)
    if region == "US":
        # Placeholder: map to sector based on first letter
        industry_id = f"sector_{ticker[0]}"
    else:
        # CN: Use first 2 digits as industry proxy
        industry_id = f"swL1_{ticker[:2]}"

    logger.debug(f"{ticker}: Assigned industry {industry_id}")
    return industry_id


def build_universe_at_date(
    recon_date: pd.Timestamp,
    all_tickers: list[str],
    instruments_dir: Path,
    config: dict,
    industry_cache: dict,
) -> pd.DataFrame:
    """
    Construct universe at a single reconstitution date.

    Steps:
    1. Load all ticker data
    2. Apply eligibility filters (price, volume)
       - Note: For CN region, ST (Special Treatment) stock filtering happens
         during data ingestion (see src.data.ingest_ts_cn.get_cn_universe_tickers),
         so only non-ST stocks are present in the instruments directory
    3. Assign industries (frozen-at-first-seen)
    4. Rank by market cap proxy (or liquidity)
    5. Select top N=800

    Args:
        recon_date: Reconstitution date (month-end trading day, timezone-naive)
        all_tickers: List of all tickers to consider
        instruments_dir: Directory containing ticker parquet files
        config: Configuration dict
        industry_cache: Dict mapping ticker -> industry_id (for frozen-at-first-seen)

    Returns:
        DataFrame with columns: date, symbol, industry_id, market_cap_proxy
    """
    logger = get_logger(__name__)

    region = config.get("region", "US")
    is_cn = region == "CN"
    universe_size = config.get("universe_size", 800)
    min_price = config.get("eligibility", {}).get("min_price", 3.0)
    min_adv_usd = config.get("eligibility", {}).get("min_adv_usd", 5000000.0)

    eligible_stocks = []

    for ticker in all_tickers:
        df = load_ticker_data(ticker, instruments_dir)
        if df is None:
            continue

        # Check if data exists at recon_date (both are now timezone-naive)
        if recon_date not in df.index:
            continue

        # Get price at recon_date
        price = df.loc[recon_date, "close"]
        if pd.isna(price) or price < min_price:
            continue

        # Compute ADV up to recon_date
        df_up_to_date = df.loc[:recon_date]
        adv_usd = compute_adv_usd(df_up_to_date, window=20, is_cn=is_cn)

        if adv_usd.empty or pd.isna(adv_usd.iloc[-1]) or adv_usd.iloc[-1] < min_adv_usd:
            continue

        # Get or assign industry (frozen-at-first-seen)
        if ticker not in industry_cache:
            industry_cache[ticker] = get_industry_classification(ticker, region)
        industry_id = industry_cache[ticker]

        # Compute market cap proxy (price * volume as of recon_date)
        volume = df.loc[recon_date, "volume"]
        market_cap_proxy = price * volume  # Simplified proxy for Sprint 0

        eligible_stocks.append(
            {
                "date": recon_date,
                "symbol": ticker,
                "industry_id": industry_id,
                "market_cap_proxy": market_cap_proxy,
                "price": price,
                "adv_usd": adv_usd.iloc[-1],
            }
        )

    if not eligible_stocks:
        logger.warning(f"{recon_date.date()}: No eligible stocks found")
        return pd.DataFrame()

    # Create DataFrame and rank by market cap proxy
    df_eligible = pd.DataFrame(eligible_stocks)
    df_eligible = df_eligible.sort_values("market_cap_proxy", ascending=False)

    # Select top N
    df_universe = df_eligible.head(universe_size)

    logger.info(
        f"{recon_date.date()}: Selected {len(df_universe)}/{len(df_eligible)} stocks for universe"
    )

    return df_universe[["date", "symbol", "industry_id", "market_cap_proxy"]]


def main():
    """Main entry point for universe construction."""
    parser = argparse.ArgumentParser(description="Build universe with monthly reconstitution")
    parser.add_argument("--config", required=True, help="Path to data config YAML")
    parser.add_argument("--output", required=True, help="Output path for universe parquet")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Load and normalize config
    with open(args.config) as f:
        raw_config = yaml.safe_load(f) or {}
    config = normalize_config(raw_config)

    # Setup logging
    logger = setup_logging(
        level="DEBUG" if args.debug else "INFO",
        config=config,
        global_seed=config.get("random_state", 42),
        region=config.get("region", "US"),
    )

    logger.info("Starting universe construction", extra={"config_path": args.config})

    # Extract config parameters
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    provider_uri = config.get("provider_uri", "data/qlib/us_data")
    calendar_name = config.get("calendar", "NYSE")

    # Get month-end dates
    month_ends = get_month_end_dates(start_date, end_date, calendar_name)

    # Get instruments directory
    instruments_dir = Path(provider_uri) / "instruments"
    if not instruments_dir.exists():
        logger.error(f"Instruments directory not found: {instruments_dir}")
        return 1

    # Get all available tickers
    ticker_files = list(instruments_dir.glob("*.parquet"))
    all_tickers = [f.stem for f in ticker_files]
    logger.info(f"Found {len(all_tickers)} tickers in {instruments_dir}")

    # Build universe at each month-end with frozen-at-first-seen industries
    universe_dfs = []
    industry_cache = {}  # Maintains frozen-at-first-seen mapping

    for recon_date in tqdm(month_ends, desc="Building universe"):
        df_universe = build_universe_at_date(
            recon_date=recon_date,
            all_tickers=all_tickers,
            instruments_dir=instruments_dir,
            config=config,
            industry_cache=industry_cache,
        )
        if not df_universe.empty:
            universe_dfs.append(df_universe)

    if not universe_dfs:
        logger.error("No universe data generated")
        return 1

    # Concatenate all month-end universes
    df_final = pd.concat(universe_dfs, ignore_index=True)

    # Save to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path)

    logger.info(
        f"Universe construction complete: {len(df_final)} rows, {len(month_ends)} months",
        extra={"output_path": str(output_path), "total_rows": len(df_final)},
    )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
