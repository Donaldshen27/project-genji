# Sprint 0 Implementation TODO

**Status:** Phase 1 (Config Foundation) ✅ Complete | Phase 2 (Data Pipeline) 🔴 TODO
**Remaining:** Phases 2-8
**Sprint 0 Spec Reference:** `docs/specs.md` Section 1 (lines 38-73)

---

## ✅ Phase 1: Configuration Foundation (COMPLETE)

All 7 YAML config files created in `configs/`:
- `data_us.yaml`, `data_cn.yaml` - Data ingestion params (yfinance/tushare)
- `model2_us.yaml`, `model2_cn.yaml` - Model 2 training (CPCV, 63-day embargo, seed=42)
- `optimizer.yaml` - QP parameters (risk model, costs, constraints)
- `backtest_us.yaml`, `backtest_cn.yaml` - Backtest configs (pyqlib settings)

**✅ Config keys frozen per Sprint 0 spec** (`docs/specs.md:45`)

---

## 🔴 Phase 2: Data Pipeline (TODO - All 4 Modules)

### Overview
Implement complete data ingestion, universe construction, and Qlib conversion for both US & CN regions.

**Spec Reference:** `docs/specs.md:46-51` (lines 46-51)
**Runtime Budget:** ≤ 6h per region on 8 vCPU / 32 GB RAM

---

### 2.1 `src/data/ingest_yf_us.py` (HIGH PRIORITY)
**Spec:** Download US equity data via yfinance

**Implementation Guide:**
```python
"""
US data ingestion using yfinance.

CLI: python -m src.data.ingest_yf_us --config configs/data_us.yaml [--debug]
"""

import argparse
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import yaml
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.utils.logging import setup_logging, get_logger

def normalize_config(raw_config: dict) -> dict:
    """Convert datetime.date objects to ISO strings for config hashing."""
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
    # Download S&P 500 constituents
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        return tables[0]["Symbol"].str.replace(".", "-").tolist()
    except Exception:
        # Fallback for testing
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B"]

def download_ticker_data(ticker: str, start_date: str, end_date: str, retry_attempts: int = 3):
    """Download OHLCV data for single ticker."""
    logger = get_logger(__name__)

    for attempt in range(retry_attempts):
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                actions=True,
                auto_adjust=False,  # Get raw prices
            )

            if df.empty:
                return None

            # Derive adjustment factors BEFORE normalizing columns
            # adj_factor = Adj Close / Close (cumulative for splits/dividends)
            if "Adj Close" in df.columns and "Close" in df.columns:
                df["adj_factor"] = df["Adj Close"] / df["Close"]
                df["adj_factor"] = df["adj_factor"].fillna(1.0)
            else:
                df["adj_factor"] = 1.0

            # Standardize column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            df.index.name = "date"

            return df
        except Exception as e:
            if attempt < retry_attempts - 1:
                continue
            else:
                logger.error(f"{ticker}: Failed after {retry_attempts} attempts: {e}")
                return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_config = yaml.safe_load(f) or {}
    config = normalize_config(raw_config)

    logger = setup_logging(
        level="DEBUG" if args.debug else "INFO",
        config=config,
        global_seed=42,
        region="US",
    )

    # Get tickers & download
    tickers = get_us_universe_tickers()
    # Use ThreadPoolExecutor to download in parallel
    # Save to data/qlib/us_data/instruments/*.parquet

    # TODO: Implement parallel download & save logic
    pass

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Key Points:**
- ⚠️ **DO NOT** pass `timeout` param to `yf.Ticker.history()` - API doesn't support it
- Use `normalize_config()` to handle datetime.date objects from YAML parsing
- Compute adj_factor = Adj Close / Close (NOT the inverse!)
- Save as parquet files per ticker

**Testing:**
```bash
uv run --locked python -m src.data.ingest_yf_us --config configs/data_us.yaml --debug
ls data/qlib/us_data/instruments/*.parquet  # Should contain ticker files
```

---

### 2.2 `src/data/ingest_ts_cn.py` (HIGH PRIORITY - ✅ IMPLEMENTED)
**Spec:** Download CN A-share data via Tushare Pro API

**Status:** Implementation complete with chunked pagination, rate limiting, and token authentication.

**Key Features:**
- ✅ Chunked pagination for date ranges > 1000 days (handles Tushare ~4000 row limit)
- ✅ Adjustment factors via `pro.adj_factor()` with proper merging
- ✅ ST stock filtering in `get_cn_universe_tickers()`
- ✅ Thread-safe API instances per worker (no shared state)
- ✅ Rate limiting between chunks (50ms delay)
- ✅ Retry logic with configurable attempts
- ✅ Volume/amount unit conversion (×100 shares, ×1000 CNY)

**CLI:**
```bash
uv run --locked python -m src.data.ingest_ts_cn --config configs/data_cn.yaml [--debug]
```

**Authentication:** Requires TUSHARE_TOKEN env var or `tushare_token` in config. Register at https://tushare.pro/register

**Reference:** https://tushare.pro/document/2

---

### 2.3 `src/data/build_universe.py` (HIGH PRIORITY)
**Spec:** Monthly reconstitution, N=800, price ≥ $3.00

**Core Logic:**
```python
def build_universe(config: dict) -> pd.DataFrame:
    """
    Construct universe with monthly reconstitution.

    Steps per month-end:
    1. Get last trading day of month (use pandas_market_calendars for NYSE/SSE)
    2. Apply eligibility filters:
       - price >= 3.0
       - 20-day avg daily volume >= min_adv_usd (5M per spec)
       - US: frozen-at-first-seen industry (yfinance sector)
       - CN: exclude ST/ST* stocks
    3. Rank by market cap or liquidity proxy
    4. Select top N=800

    Returns:
        DataFrame [date, symbol, industry_id, market_cap]
    """
    # TODO: Implement
    pass
```

**CLI:** `python -m src.data.build_universe --config configs/data_{region}.yaml --output data/universe_{region}.parquet`

**Dependencies:**
- `pandas_market_calendars` for trading calendars
- Industry classification: yfinance (US), tushare (CN)

---

### 2.4 `src/data/build_qlib_dataset.py` (MEDIUM PRIORITY)
**Spec:** Convert parquet to Qlib binary format

```python
import qlib

def convert_to_qlib(config: dict):
    """
    Convert raw data to Qlib binary format.

    Use qlib.data.dump_bin() to create:
    - Features: $open, $high, $low, $close, $volume, $change, $factor
    - Save to data/qlib/{region}_data/

    Reference: https://qlib.readthedocs.io/en/latest/component/data.html#dump-data
    """
    pass
```

**CLI:** `python -m src.data.build_qlib_dataset --config configs/data_{region}.yaml`

**Acceptance Test:**
```python
# tests/integration/test_data_smoke.py
def test_qlib_init():
    import qlib
    qlib.init(provider_uri="data/qlib/us_data", region="US")
    qlib.init(provider_uri="data/qlib/cn_data", region="CN")
```

---

### 2.5 Update Makefile (✅ COMPLETE)
**File:** `Makefile:64-76`

**Status:** Makefile updated with proper commands:
```makefile
data_us:
	@echo "Running US data pipeline..."
	uv run --locked python -m src.data.ingest_yf_us --config configs/data_us.yaml
	uv run --locked python -m src.data.build_universe --config configs/data_us.yaml --output data/universe_us.parquet
	uv run --locked python -m src.data.build_qlib_dataset --config configs/data_us.yaml
	@echo "US data pipeline complete"

data_cn:
	@echo "Running CN data pipeline..."
	uv run --locked python -m src.data.ingest_ts_cn --config configs/data_cn.yaml
	uv run --locked python -m src.data.build_universe --config configs/data_cn.yaml --output data/universe_cn.parquet
	uv run --locked python -m src.data.build_qlib_dataset --config configs/data_cn.yaml
	@echo "CN data pipeline complete"
```

**Usage:**
```bash
make data_us  # Run US data pipeline (ingest → universe → qlib)
make data_cn  # Run CN data pipeline (ingest → universe → qlib)
```

---

## 🔴 Phase 3-8: See Full Details Below

*(Phases 3-8 continue as before with Model 2, Optimizer, Backtest, Tests, Makefile, Validation)*

---

## 📊 Quick Effort Estimates

| Phase | Estimated Effort |
|-------|------------------|
| 2 - Data Pipeline | 2-3 days |
| 3 - Model 2 | 4-5 days |
| 4 - Optimizer | 3-4 days |
| 5 - Backtest | 2-3 days |
| 6 - Tests | 2-3 days |
| 7 - Makefile | 0.5 days |
| 8 - Validation | 1 day |
| **Total** | **15-20 days** |

---

## 🎯 Recommended Start Order

1. **Implement** `src/data/ingest_yf_us.py` (4-6 hrs)
2. ✅ **DONE** `src/data/ingest_ts_cn.py` - Tushare ingestion with pagination
3. **Implement** `src/data/build_universe.py` (6-8 hrs)
4. **Implement** `src/data/build_qlib_dataset.py` (4-6 hrs)
5. **Update** Makefile data targets (15 min)
6. **Test:** `make data_us && make data_cn`
7. **Move to Phase 3** (Model 2)

---

## ⚠️ Critical Constraints

1. **63-day embargo** in CPCV (Model 2)
2. **Seed=42** for all random operations
3. **No new config keys** in Sprint 0
4. **Coverage >= 80%** enforced
5. **Industry caps within 1e-6**
6. **QP is single source of truth** for weights

---

## 🔗 Key References

- **Specs:** `docs/specs.md` Section 1 (Sprint 0)
- **Theory:** `docs/theory.md`
- **Project Guide:** `CLAUDE.md`
- **Qlib:** https://qlib.readthedocs.io/
- **CVXPY:** https://www.cvxpy.org/
