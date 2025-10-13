"""
US equity data ingestion using yfinance.

CLI: python -m src.data.ingest_yf_us --config configs/data_us.yaml [--debug]
"""

import argparse
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

from src.utils.logging import get_logger, setup_logging


def normalize_config(raw_config: dict) -> dict:
    """
    Convert datetime.date objects to ISO strings for config hashing.

    YAML's safe_load() converts ISO dates (e.g., 2003-01-01) to datetime.date objects.
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


def get_us_universe_tickers() -> list[str]:
    """
    Get US equity tickers for universe.

    For Sprint 0, use S&P 500 as proxy. Production should use full universe
    from data vendor (CRSP, Compustat).
    """
    logger = get_logger(__name__)

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # Add User-Agent header to avoid 403 Forbidden from Wikipedia
        # Using urllib.request (stdlib) to avoid adding fsspec dependency
        req = Request(url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

        with urlopen(req, timeout=10) as response:
            # Decode bytes to string for pd.read_html
            charset = response.headers.get_content_charset("utf-8") or "utf-8"
            html_text = response.read().decode(charset, errors="ignore")

        tables = pd.read_html(io.StringIO(html_text))
        tickers = tables[0]["Symbol"].str.replace(".", "-").tolist()
        logger.info(f"Retrieved {len(tickers)} tickers from S&P 500")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list: {e}. Using fallback tickers.")
        # Fallback for testing
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B"]


def download_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    retry_attempts: int = 3,
) -> pd.DataFrame | None:
    """
    Download OHLCV data for single ticker.

    Args:
        ticker: Stock symbol
        start_date: Start date (inclusive) as ISO string
        end_date: End date (inclusive) as ISO string
        retry_attempts: Number of retry attempts on failure

    Returns DataFrame with columns: date, open, high, low, close, volume, adj_close, adj_factor
    or None if download fails.

    Note: yfinance's history() uses exclusive end date, so we add one day internally.
    """
    logger = get_logger(__name__)

    # yfinance end parameter is exclusive, so add one day to include the configured end_date
    if end_date:
        end_date_obj = datetime.fromisoformat(end_date)
        end_date_inclusive = (end_date_obj + timedelta(days=1)).isoformat()[:10]
    else:
        end_date_inclusive = None

    for attempt in range(retry_attempts):
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date_inclusive,  # Now inclusive of configured end_date
                actions=True,
                auto_adjust=False,  # Get raw prices
                # NOTE: DO NOT use timeout parameter - API doesn't support it
            )

            if df.empty:
                logger.warning(f"{ticker}: No data returned")
                return None

            # Derive adjustment factors BEFORE normalizing columns
            # adj_factor = Adj Close / Close (cumulative for splits/dividends)
            # For 2-for-1 split: Close=$100 → $50, Adj Close=$100 → $50, factor=1.0 (no change)
            # For dividend $2: Close=$100 (unchanged), Adj Close adjusts, factor=(Adj/Close)
            if "Adj Close" in df.columns and "Close" in df.columns:
                df["adj_factor"] = df["Adj Close"] / df["Close"]
                df["adj_factor"] = df["adj_factor"].fillna(1.0)
            else:
                logger.warning(f"{ticker}: Missing Adj Close or Close, setting adj_factor=1.0")
                df["adj_factor"] = 1.0

            # Standardize column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            df.index.name = "date"

            # Keep only required fields
            required_cols = ["open", "high", "low", "close", "volume", "adj_close", "adj_factor"]
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols]

            logger.debug(f"{ticker}: Downloaded {len(df)} rows")
            return df

        except Exception as e:
            if attempt < retry_attempts - 1:
                logger.debug(f"{ticker}: Attempt {attempt + 1} failed: {e}, retrying...")
                continue
            else:
                logger.error(f"{ticker}: Failed after {retry_attempts} attempts: {e}")
                return None


def save_ticker_data(ticker: str, df: pd.DataFrame, output_dir: Path) -> None:
    """Save ticker data to parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.parquet"
    df.to_parquet(output_path)


def main():
    """Main entry point for US data ingestion."""
    parser = argparse.ArgumentParser(description="Download US equity data via yfinance")
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
        region=config.get("region", "US"),
    )

    logger.info("Starting US data ingestion", extra={"config_path": args.config})

    # Extract config parameters
    start_date = config.get("start_date")
    end_date = config.get("end_date")
    provider_uri = config.get("provider_uri", "data/qlib/us_data")
    max_workers = config.get("max_workers", 8)  # Fixed: use correct config key
    retry_attempts = config.get("retry_attempts", 3)  # Fixed: read from config

    # Get tickers
    tickers = get_us_universe_tickers()
    logger.info(f"Downloading data for {len(tickers)} tickers")

    # Setup output directory
    output_dir = Path(provider_uri) / "instruments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks with configurable retry attempts
        future_to_ticker = {
            executor.submit(
                download_ticker_data, ticker, start_date, end_date, retry_attempts
            ): ticker
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
        extra={"successful": successful, "failed": failed, "total": len(tickers)},
    )

    if successful == 0:
        logger.error("No data downloaded successfully")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
