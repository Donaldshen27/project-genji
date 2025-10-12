"""
Convert raw parquet data to Qlib binary format.

Implements conversion from instruments/*.parquet to Qlib binary format
with required features: $open, $high, $low, $close, $volume, $change, $factor.

Uses DumpDataAll from dump_bin module to create proper .bin artifacts.

CLI: python -m src.data.build_qlib_dataset --config configs/data_{region}.yaml
"""

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as loguru_logger

from src.utils.logging import setup_logging
from src.data.dump_bin import DumpDataAll


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


class QlibDumper(DumpDataAll):
    """
    Custom dumper that preprocesses data to add required Qlib fields.

    Overrides _get_source_data to:
    - Validate required OHLCV columns exist
    - Sort data by date chronologically
    - Compute '$change' field (daily returns)
    - Rename 'adj_factor' to '$factor'
    - Add '$' prefix to all feature columns (Qlib convention)
    - Ensure 'date' is a regular column (not index)
    """

    def _get_source_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load and preprocess ticker data for Qlib format.

        Validates:
        - Required OHLCV columns exist (open, high, low, close, volume)

        Adds:
        - $change: daily returns (close.pct_change() after sorting by date)
        - $factor: renamed from adj_factor

        All feature columns are prefixed with $ per Qlib convention.

        Returns:
            DataFrame with columns: date, $open, $high, $low, $close, $volume, $change, $factor
            or empty DataFrame if required columns are missing
        """
        # Read parquet file
        df = pd.read_parquet(file_path)

        # Ensure date is a regular column (not index)
        if 'date' not in df.columns:
            # If date is in index, reset it to a column
            df = df.reset_index()
            # If the index had a different name or no name, rename the first column to 'date'
            if 'date' not in df.columns:
                # After reset_index, the old index becomes a column
                # If it wasn't named 'date', rename it now
                if df.index.name is not None:
                    # The old index name became a column
                    df = df.rename(columns={df.columns[0]: 'date'})
                else:
                    # Unnamed index became 'index' column, rename it
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'date'})
                    else:
                        df = df.rename(columns={df.columns[0]: 'date'})

        # Ensure date column is datetime and timezone-naive
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

        # Validate required OHLCV columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            loguru_logger.warning(
                f"{file_path.stem}: Missing required columns {missing_cols}, skipping"
            )
            return pd.DataFrame()  # Return empty DataFrame to skip this ticker

        # Sort by date to ensure chronological order before computing returns
        df = df.sort_values('date').reset_index(drop=True)

        # Compute change (daily returns) from close prices (AFTER sorting)
        df['$change'] = df['close'].pct_change().fillna(0.0)

        # Rename adj_factor to $factor (Qlib expects '$factor')
        if 'adj_factor' in df.columns:
            df['$factor'] = df['adj_factor']
            df = df.drop(columns=['adj_factor'])
        else:
            loguru_logger.warning(f"{file_path.stem}: Missing adj_factor, using 1.0")
            df['$factor'] = 1.0

        # Add $ prefix to OHLCV columns (Qlib convention)
        # Map: open → $open, high → $high, low → $low, close → $close, volume → $volume
        rename_map = {
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'close': '$close',
            'volume': '$volume',
        }
        df = df.rename(columns=rename_map)

        # Drop adj_close (not needed for Qlib)
        if 'adj_close' in df.columns:
            df = df.drop(columns=['adj_close'])

        return df


def main():
    """Main entry point for Qlib dataset conversion."""
    parser = argparse.ArgumentParser(description="Convert raw data to Qlib binary format")
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

    logger.info("Starting Qlib dataset conversion", extra={"config_path": args.config})

    # Extract config parameters
    provider_uri = config.get("provider_uri", "data/qlib/us_data")
    max_workers = config.get("max_workers", 8)

    # Input: instruments directory with raw parquet files
    instruments_dir = Path(provider_uri) / "instruments"
    if not instruments_dir.exists():
        logger.error(f"Instruments directory not found: {instruments_dir}")
        return 1

    # Output: Qlib binary format in provider_uri root
    output_dir = Path(provider_uri)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting data from {instruments_dir} to Qlib format at {output_dir}")

    # Fields to include in dump (with $ prefix as preprocessed by QlibDumper._get_source_data)
    # These match the exact Qlib convention: $open, $high, $low, $close, $volume, $change, $factor
    include_fields = "$open,$high,$low,$close,$volume,$change,$factor"

    try:
        # Create QlibDumper instance and run conversion
        dumper = QlibDumper(
            data_path=str(instruments_dir),
            qlib_dir=str(output_dir),
            freq="day",
            max_workers=max_workers,
            date_field_name="date",
            file_suffix=".parquet",
            symbol_field_name="symbol",
            include_fields=include_fields,
            exclude_fields="",  # No exclusions needed, preprocessing handles it
        )

        logger.info(f"Running QlibDumper with {len(dumper.df_files)} files")
        dumper.dump()

        logger.info("Qlib dataset conversion complete")
        return 0

    except Exception as e:
        logger.error(f"Qlib conversion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
