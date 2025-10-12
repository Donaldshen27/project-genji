"""
CN A-share data ingestion using Tushare API.

CLI: python -m src.data.ingest_ts_cn --config configs/data_cn.yaml [--debug]

Note: Requires Tushare API token. Set via environment variable TUSHARE_TOKEN
or pass in config file.
"""

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
import tushare as ts
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.utils.logging import setup_logging, get_logger


def normalize_config(raw_config: dict) -> dict:
    """
    Convert datetime.date objects to ISO strings for config hashing.

    YAML's safe_load() converts ISO dates (e.g., 2010-01-01) to datetime.date objects.
    This breaks compute_config_hash() which requires JSON-serializable types.
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


def get_tushare_api(token: Optional[str] = None):
    """
    Initialize Tushare Pro API with token.

    Args:
        token: Tushare API token. If None, reads from TUSHARE_TOKEN env var.

    Returns:
        Tushare Pro API instance
    """
    logger = get_logger(__name__)

    if token is None:
        token = os.environ.get("TUSHARE_TOKEN")

    if not token:
        raise ValueError(
            "Tushare API token required. Set TUSHARE_TOKEN environment variable "
            "or provide 'tushare_token' in config file."
        )

    try:
        ts.set_token(token)
        pro = ts.pro_api()
        logger.info("Tushare API initialized successfully")
        return pro
    except Exception as e:
        logger.error(f"Failed to initialize Tushare API: {e}")
        raise


def get_cn_universe_tickers(token: str) -> list[str]:
    """
    Get CN A-share stock codes for universe.

    For Sprint 0, use all A-share stocks. Production should filter based on
    liquidity and eligibility criteria.

    Args:
        token: Tushare API token (creates its own API instance to avoid thread safety issues)

    Returns:
        List of Tushare stock codes (e.g., "000001.SZ", "600000.SH")
    """
    logger = get_logger(__name__)

    try:
        # Create a dedicated API instance for this call
        pro = get_tushare_api(token)

        # Get all A-share stock list
        # list_status: L=listed, D=delisted, P=paused
        stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')

        # Filter out stocks with "ST" in name (special treatment stocks)
        stock_info = stock_info[~stock_info["name"].str.contains("ST", na=False)]
        tickers = stock_info["ts_code"].tolist()

        logger.info(f"Retrieved {len(tickers)} tickers from CN A-shares (ST stocks excluded)")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch CN A-share list: {e}. Using fallback tickers.")
        # Fallback for testing (major CN stocks)
        return [
            "000001.SZ",  # Ping An Bank
            "000002.SZ",  # Vanke
            "600000.SH",  # Pudong Development Bank
            "600519.SH",  # Kweichow Moutai
            "600036.SH",  # China Merchants Bank
            "000858.SZ",  # Wuliangye
            "601318.SH",  # Ping An Insurance
            "601398.SH",  # ICBC
        ]


def chunk_date_range(start_date: str, end_date: str, chunk_days: int = 365) -> list[tuple[str, str]]:
    """
    Split date range into smaller chunks to handle Tushare pagination limits.

    Tushare API limits responses to ~4000 rows per request. By chunking the date range,
    we ensure we get complete data.

    Args:
        start_date: Start date as ISO string "YYYY-MM-DD"
        end_date: End date as ISO string "YYYY-MM-DD"
        chunk_days: Number of days per chunk (default 365)

    Returns:
        List of (start, end) date tuples as ISO strings
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    chunks = []
    current = start

    # Use <= to handle single-day ranges and ensure end date is included
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))
        current = chunk_end + timedelta(days=1)

    return chunks


def download_ticker_data(
    token: str,
    ticker: str,
    start_date: str,
    end_date: str,
    retry_attempts: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for single CN stock using Tushare API.

    Args:
        token: Tushare API token (creates its own API instance to avoid thread safety issues)
        ticker: Stock code in Tushare format (e.g., "000001.SZ")
        start_date: Start date (inclusive) as ISO string "YYYY-MM-DD"
        end_date: End date (inclusive) as ISO string "YYYY-MM-DD"
        retry_attempts: Number of retry attempts on failure

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount, adj_factor
        or None if download fails.

    Note: Tushare provides both raw and adjusted prices via adj_factor field.
    """
    logger = get_logger(__name__)

    # Create a dedicated API instance for this thread to avoid concurrency issues
    pro = get_tushare_api(token)

    # Split date range into chunks to handle Tushare's row limit (~4000 per request)
    date_chunks = chunk_date_range(start_date, end_date, chunk_days=1000)

    all_dfs = []
    all_adj_dfs = []

    for chunk_start, chunk_end in date_chunks:
        # Tushare expects date format "YYYYMMDD"
        start_date_ts = chunk_start.replace("-", "")
        end_date_ts = chunk_end.replace("-", "")

        for attempt in range(retry_attempts):
            try:
                # Download daily data
                df = pro.daily(
                    ts_code=ticker,
                    start_date=start_date_ts,
                    end_date=end_date_ts,
                )

                if df is not None and not df.empty:
                    all_dfs.append(df)

                # Get adjustment factors
                df_adj = pro.adj_factor(
                    ts_code=ticker,
                    start_date=start_date_ts,
                    end_date=end_date_ts,
                )

                if df_adj is not None and not df_adj.empty:
                    all_adj_dfs.append(df_adj)

                # Success - break retry loop
                break

            except Exception as e:
                if attempt < retry_attempts - 1:
                    logger.debug(
                        f"{ticker}: Attempt {attempt + 1} failed for chunk "
                        f"{chunk_start} to {chunk_end}: {e}, retrying..."
                    )
                    time.sleep(0.1)  # Brief pause before retry
                    continue
                else:
                    logger.error(
                        f"{ticker}: Failed after {retry_attempts} attempts for chunk "
                        f"{chunk_start} to {chunk_end}: {e}"
                    )
                    # Continue to next chunk even if this one failed
                    break

        # Brief pause between chunks to respect API rate limits
        time.sleep(0.05)

    # Check if we got any data
    if not all_dfs:
        logger.warning(f"{ticker}: No data returned from any chunk")
        return None

    # Concatenate all chunks
    df = pd.concat(all_dfs, ignore_index=True)

    # Merge adjustment factors if available
    if all_adj_dfs:
        df_adj = pd.concat(all_adj_dfs, ignore_index=True)
        df = df.merge(df_adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
        df['adj_factor'] = df['adj_factor'].fillna(1.0)
    else:
        logger.warning(f"{ticker}: No adjustment factors available, using 1.0")
        df['adj_factor'] = 1.0

    # Tushare column mapping:
    # trade_date -> date
    # open, high, low, close (raw prices)
    # vol -> volume (in shares, need to multiply by 100 for Tushare data)
    # amount -> amount (in 1000 CNY, need to multiply by 1000)
    df = df.rename(columns={
        'trade_date': 'date',
        'vol': 'volume',
    })

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.set_index('date')

    # Remove duplicates (in case chunks overlapped)
    df = df[~df.index.duplicated(keep='first')]

    # Sort by date (ascending)
    df = df.sort_index()

    # Tushare volume is in 100 shares, convert to shares
    df['volume'] = df['volume'] * 100

    # Tushare amount is in 1000 CNY, convert to CNY
    df['amount'] = df['amount'] * 1000

    # Compute adjusted close: adj_close = close * adj_factor
    df['adj_close'] = df['close'] * df['adj_factor']

    # Keep only required fields
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'adj_close', 'adj_factor']
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]

    logger.debug(f"{ticker}: Downloaded {len(df)} rows from {len(date_chunks)} chunks")
    return df


def save_ticker_data(ticker: str, df: pd.DataFrame, output_dir: Path) -> None:
    """Save ticker data to parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use simplified ticker name for filename (remove .SZ/.SH suffix)
    ticker_name = ticker.split('.')[0]
    output_path = output_dir / f"{ticker_name}.parquet"
    df.to_parquet(output_path)


def main():
    """Main entry point for CN data ingestion."""
    parser = argparse.ArgumentParser(description="Download CN A-share data via Tushare")
    parser.add_argument("--config", required=True, help="Path to data config YAML")
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
        region=config.get("region", "CN"),
    )

    logger.info("Starting CN data ingestion", extra={"config_path": args.config})

    # Get Tushare API token
    token = config.get("tushare_token") or os.environ.get("TUSHARE_TOKEN")
    if not token:
        logger.error(
            "Tushare API token required. Set TUSHARE_TOKEN environment variable "
            "or provide 'tushare_token' in config file."
        )
        return 1

    # Extract and validate config parameters
    start_date = config.get("start_date")
    end_date = config.get("end_date")

    # Validate required date parameters
    if not start_date or not end_date:
        logger.error(
            "Missing required date parameters in config. "
            "Both 'start_date' and 'end_date' must be provided as ISO strings (YYYY-MM-DD)."
        )
        return 1

    # Ensure dates are strings (normalize_config should have handled datetime objects)
    if not isinstance(start_date, str) or not isinstance(end_date, str):
        logger.error(
            f"Invalid date format: start_date={start_date}, end_date={end_date}. "
            "Expected ISO strings (YYYY-MM-DD)."
        )
        return 1

    provider_uri = config.get("provider_uri", "data/qlib/cn_data")
    max_workers = config.get("max_workers", 4)  # Lower default to respect API rate limits
    retry_attempts = config.get("retry_attempts", 3)

    logger.info(
        f"Data ingestion parameters: start_date={start_date}, end_date={end_date}, "
        f"max_workers={max_workers}, retry_attempts={retry_attempts}"
    )

    # Get tickers
    tickers = get_cn_universe_tickers(token)
    logger.info(f"Downloading data for {len(tickers)} tickers")

    # Setup output directory
    output_dir = Path(provider_uri) / "instruments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks with configurable retry attempts
        # Pass token instead of shared pro instance to avoid thread safety issues
        future_to_ticker = {
            executor.submit(download_ticker_data, token, ticker, start_date, end_date, retry_attempts): ticker
            for ticker in tickers
        }

        # Process completed downloads with progress bar
        for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Downloading"):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    save_ticker_data(ticker, df, output_dir)
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"{ticker}: Unexpected error during processing: {e}")
                failed += 1

    logger.info(
        f"Download complete: {successful} successful, {failed} failed",
        extra={"successful": successful, "failed": failed, "total": len(tickers)}
    )

    if successful == 0:
        logger.error("No data downloaded successfully")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
