# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Sprint 0 - Baseline Implementation (In Progress)
- Model 2 (stock alpha forecasting)
- QP optimizer with risk/cost model
- Qlib backtest integration
- HTML report generation
- Test suite (unit + integration)

## [0.2.0] - 2025-10-12

### Added - Sprint 0 Phase 2: Data Pipeline

#### Data Ingestion
- **US Market (`src/data/ingest_yf_us.py`)**:
  - yfinance-based OHLCV data ingestion for S&P 500 constituents
  - Parallel download with ThreadPoolExecutor (configurable workers)
  - Retry logic with configurable attempts
  - Adjustment factor computation (Adj Close / Close)
  - Parquet file storage per ticker
  - Wikipedia-based S&P 500 ticker list with fallback

- **CN Market (`src/data/ingest_ts_cn.py`)**:
  - Tushare Pro API integration for A-share data
  - Chunked pagination for date ranges > 1000 days (handles API ~4000 row limit)
  - Rate limiting between chunks (50ms delay)
  - Thread-safe API instances per worker
  - Adjustment factors via `pro.adj_factor()` with proper merging
  - Volume/amount unit conversion (×100 shares, ×1000 CNY)
  - ST stock filtering support
  - Token authentication via environment variable or config

#### Universe Construction (`src/data/build_universe.py`)
- Monthly reconstitution at last trading day per region calendar
- Eligibility filters: price ≥ $3.00, minimum 20-day ADV
- Top N=800 selection by market cap proxy (close × volume)
- Frozen-at-first-seen industry assignment (placeholder for Sprint 0)
- pandas-market-calendars integration for NYSE/SSE trading days
- Handles timezone-aware/naive datetime conversions

#### Qlib Dataset Conversion (`src/data/build_qlib_dataset.py`)
- Conversion from parquet to Qlib binary format
- Integration with dump_bin module (`src/data/dump_bin.py`)
- Required features: $open, $high, $low, $close, $volume, $change, $factor
- Proper calendar and feature file generation
- Handles missing adjustment factors (defaults to 1.0)

#### Configuration
- `configs/data_us.yaml`: US data ingestion parameters
- `configs/data_cn.yaml`: CN data ingestion parameters with Tushare token support

#### Makefile Targets
- `make data_us`: Full US data pipeline (ingest → universe → qlib)
- `make data_cn`: Full CN data pipeline (ingest → universe → qlib)

### Technical Details
- Parallel processing for data ingestion (ThreadPoolExecutor)
- Robust error handling and retry mechanisms
- Comprehensive logging with progress bars (tqdm)
- Config normalization for datetime.date objects
- Deterministic config hashing integration

### Dependencies
- pandas-market-calendars >= 4.0.0 (trading calendar management)
- pyarrow >= 14.0.0 (Parquet support)
- tushare >= 1.4.0 (CN market data)
- yfinance >= 0.2.0 (US market data)

## [0.1.0] - 2025-10-12

### Added - Section 0: Global Foundation

#### Project Structure
- Python 3.11 project with uv package manager
- Complete directory skeleton (configs, src, data, tests, reports, scripts)
- Package configuration with hatchling build backend
- 275 dependencies resolved and locked in `uv.lock`

#### Core Dependencies
- **Data & Backtesting**: pyqlib ≥0.9.5
- **Optimization**: cvxpy, OSQP, Clarabel
- **Machine Learning**: scikit-learn, xgboost
- **Deep Learning** (optional): torch, stable-baselines3, gymnasium
- **Statistical Models**: statsmodels, hmmlearn, arch (GARCH/DCC)
- **Data Sources**: yfinance, tushare

#### Logging Infrastructure (`src/utils/logging.py`)
- JSON line format with UTC timestamps
- Automatic metadata injection: `git_sha`, `config_hash`, `global_seed`, `region`, `split_window_ids`
- Deterministic config hashing with type normalization
- Configurable log levels (INFO default, DEBUG on demand)

#### CI/CD Pipeline (`.github/workflows/ci.yml`)
- Automated testing on push/PR to main
- Uses `astral-sh/setup-uv@v5` for reproducible environments
- Enforces code quality: ruff formatting + linting
- Test coverage ≥80% threshold with pytest
- Coverage reports uploaded to Codecov

#### Development Tools
- **Makefile** with commands:
  - `make install` - Install dependencies with locked versions
  - `make test` / `make test-cov` - Run tests with coverage
  - `make lint` / `make format` - Code quality checks
  - `make clean` - Remove build artifacts
  - Sprint 0 pipeline stubs: `data_us`, `data_cn`, `train_us/cn`, `opt_US/CN`, `backtest_us/cn`, `report_US/CN`, `e2e_us/cn`

#### Configuration Files
- `.gitignore` - Comprehensive ignore patterns (data, models, reports, secrets)
- `.python-version` - Python 3.11 pinned
- `pyproject.toml` - Complete project metadata with tool configs (ruff, pytest, coverage)

#### Documentation
- `README.md` - Project overview with accurate status (early Sprint 0)
- `CHANGELOG.md` - This file
- `docs/specs.md` - Complete technical specifications (all sprints)
- `docs/theory.md` - Mathematical foundations

#### Module Stubs
- `src/backtest/` - Backtesting engine (placeholder)
- `src/data/` - Data ingestion (placeholder)
- `src/model1/` - Industry regime models (placeholder)
- `src/model2/` - Stock alpha models (placeholder)
- `src/optimizer/` - QP optimizer (placeholder)
- `src/rl/` - RL controller (placeholder)
- `src/utils/` - Shared utilities (logging ✅ implemented)
- `tests/unit/` - Unit tests (skeleton)
- `tests/integration/` - Integration tests (skeleton)

### Technical Details

#### Reproducibility Guarantees
- Global seed infrastructure (42 default, logged in metadata)
- Deterministic config hashing (SHA256, first 16 chars)
- Locked dependencies via `uv.lock` (committed)
- Git SHA tracking in all logs

#### Code Quality Standards
- Line length: 100 characters
- Python target: 3.11
- Linting: ruff with pycodestyle, pyflakes, isort, flake8-bugbear, comprehensions, pyupgrade
- Test coverage: ≥80% enforced across `src/`

---

## Release Notes

### [0.1.0] - Foundation Release
This release establishes the zero-ambiguity foundation for the quant-system project, implementing all global invariants from Section 0 of the specifications. The repository is now ready for Sprint 0 implementation (baseline model + backtest pipeline).

**No breaking changes** - this is the initial release.

---

[Unreleased]: https://github.com/yourusername/quant-system/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/quant-system/releases/tag/v0.1.0
