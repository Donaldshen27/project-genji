# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quant-system** is an industry-aware equity trading system implementing signal/decision separation across Model 1 (regime detection), Model 2 (stock alpha), a QP optimizer, and an optional RL controller. The system supports US (yfinance) and CN (Tushare) markets with deterministic backtesting via pyqlib.

**Current Status:** Sprint 0 foundation complete; core pipeline modules (data ingestion, models, optimizer, backtest) are stubs awaiting implementation.

## Essential Commands

### Setup
```bash
# Install all dependencies (always use locked mode)
make install
# or: uv sync --locked --all-extras --dev
```

### Development Workflow
```bash
make format         # Format with ruff (auto-fix)
make lint           # Check with ruff
make test           # Fast test run
make test-cov       # Full coverage report (≥80% required)
make clean          # Remove artifacts
```

### Running Modules (Sprint 0+ CLI pattern)
```bash
# Always use: uv run --locked python -m <module> --config <config>
uv run --locked python -m src.data.ingest_yf_us --config configs/data_us.yaml
uv run --locked python -m src.model2.train --config configs/model2_us.yaml
uv run --locked python -m src.optimizer.solve_qp --config configs/optimizer.yaml --region US
uv run --locked python -m src.backtest.run_backtest --config configs/backtest_us.yaml
```

### End-to-End Pipelines
```bash
make e2e_us         # US: data → train → optimize → backtest → report
make e2e_cn         # CN: data → train → optimize → backtest → report
```

## Architecture Principles

### Signal/Decision Separation (Non-Negotiable)
1. **Model 2** (stock alpha) outputs expected returns → feeds optimizer
2. **Model 1** (industry regimes, S1+) outputs tilts/caps/vol-target → feeds optimizer
3. **Optimizer** (QP) is the **single source of truth** for portfolio weights
4. **RL Controller** (S3+) tunes optimizer hyperparameters within safe bounds; **never bypasses QP**

### Two-Region Architecture
- **US:** yfinance data, frozen-at-first-seen industry assignment
- **CN:** Tushare data (A-shares), point-in-time (PIT) industry membership
- Universe: N=800 stocks, monthly reconstitution, price ≥ $3.00 eligibility
- Configs must be region-specific: `configs/data_{us,cn}.yaml`, `configs/model2_{region}.yaml`, etc.

### Data Flow
```
Raw prices (yfinance/Tushare)
  → Universe construction (monthly, N=800)
    → Qlib datasets (data/qlib/{us_data,cn_data})
      → Model 2 training (CPCV, 63-day embargo)
        → Alpha predictions
          → QP optimizer (cvxpy + OSQP/Clarabel)
            → Portfolio weights
              → Qlib backtest
                → HTML reports
```

## Critical Invariants

### Reproducibility (Global Seed = 42)
- **All model training** must accept `random_state=42` via config
- **Logging** automatically injects: `git_sha`, `config_hash`, `global_seed`, `region`, `split_window_ids`
- Use `src.utils.logging.setup_logging()` for JSON-line logs with UTC timestamps
- Config hashing: `src.utils.logging.compute_config_hash()` deterministically hashes YAML dicts

### Dependency Management
- **Never** run `uv sync` without `--locked` in scripts/CI
- **Never** modify `pyproject.toml` dependencies without regenerating `uv.lock`
- Optional groups: `[rl]` for torch/stable-baselines3, `[dev]` for testing tools

### Test Coverage
- **Minimum 80%** across `src/` (enforced in CI)
- Run `make test-cov` before committing
- Coverage config: `pyproject.toml` → `[tool.coverage.run]`

### Code Quality
- Line length: **100 chars**
- Linter: `ruff` with pycodestyle, pyflakes, isort, flake8-bugbear, pyupgrade
- Auto-format before committing: `make format`

## Module Structure & Responsibilities

### `src/data/`
- **Ingestion:** `ingest_yf_us.py`, `ingest_ts_cn.py` → raw price/volume to `data/qlib/`
- **Universe:** `build_universe.py` → N=800 monthly recon, eligibility filters
- **PIT Industries (S1+):** `industries_cn_pit.py` → Tushare snapshots for CN

### `src/model2/`
- Stock-level alpha forecasting
- Must use **CPCV with 63-day embargo**
- Outputs: expected returns per stock per day
- CLI: `python -m src.model2.train --config configs/model2_{region}.yaml`

### `src/model1/` (Sprint 1+)
- **Per-industry regime detection:** HMM-Trend (3-state), HMM-Vol (3-state), BOCPD changepoints, DCC(1,1) correlations
- **Mapping:** regime probabilities → industry tilts (bp), exposure caps, vol-target scaling
- CLI: `python -m src.model1.run --config configs/model1_{region}.yaml --out data/model1/{region}/states.parquet`
- Output schema: `date, industry_id, p_trend_{bull,bear,sideways}, p_vol_{calm,turbulent,crisis}, p_sys_{diversifying,systemic}, p_flip_10d, tilt_bp, cap_long, cap_short`

### `src/optimizer/`
- **QP solver:** cvxpy with OSQP or Clarabel backend
- **Risk model:** Ledoit-Wolf covariance shrinkage (S0 baseline)
- **Cost model:** L1 (turnover) + L2 (impact); liquidity-bucketed in S2+
- **Inputs:** alphas from Model 2, tilts/caps from Model 1 (S1+), cost parameters
- **Output:** portfolio weights
- CLI: `python -m src.optimizer.solve_qp --config configs/optimizer.yaml --region {US|CN}`

### `src/backtest/`
- **Engine:** pyqlib backtester with custom executor
- **Execution (S2+):** POV/VWAP scheduling, ADV caps, min trade notional
- **Reporting:** HTML reports with Sharpe/DSR, max DD, turnover, costs, industry exposures
- CLI: `python -m src.backtest.run_backtest --config configs/backtest_{region}.yaml`
- CLI: `python -m src.backtest.report --region {US|CN} --out reports/sprint0_{REGION}/report.html`

### `src/rl/` (Sprint 3+)
- **PPO controller** for tuning optimizer hyperparameters: γ, λ₁, λ₂, τ, caps multiplier, vol-target
- **Safe action bounds:** γ ∈ [4.0, 20.0], λ₁ ∈ [1e-4, 2e-3], etc. (see specs)
- **Training:** market replay 2010-2018, validation 2019, test 2020-2025
- CLI: `python -m src.rl.train --config configs/rl.yaml --region {US|CN}`
- CLI: `python -m src.rl.eval --config configs/rl.yaml --region {US|CN}`

### `src/utils/`
- **`logging.py`:** JSON-line logging with deterministic config hashing ✅ (implemented)
- **Future:** RNG seeding utility, metrics computation (Sharpe, DSR), helpers

## Configuration Conventions

### File Naming
- Region-specific: `configs/{data,model2,backtest}_{us,cn}.yaml`
- Shared: `configs/optimizer.yaml`, `configs/rl.yaml` (S3+)
- Model 1 (S1+): `configs/model1_{region}.yaml`

### Required Keys (per Sprint 0 spec)
- **All configs** must include: `random_state: 42`, `region: {US|CN}`
- **Data configs:** `provider_uri`, `start_date`, `end_date`, `universe_size`, `eligibility`
- **Model configs:** `features`, `labels`, `cv_scheme` (CPCV with `embargo_days: 63`)
- **Optimizer config:** `risk_model`, `cost_model`, `constraints` (turnover, vol_target, industry_caps)
- **Backtest config:** `executor_type`, `slippage_model`, `report_metrics`

### Where Configs Live
- Development: `configs/` (version controlled)
- Production runs (S4): immutable bundles in `configs/bundles/<date>_<git_sha>/`

## Data Conventions

### Storage Locations
- **Qlib datasets:** `data/qlib/{us_data,cn_data}/` (git-ignored)
- **Model 1 outputs (S1+):** `data/model1/{region}/states.parquet`
- **Cost params (S2+):** `data/costs/{region}/lambda_params.parquet`
- **Reports:** `reports/<sprint>_<region>/` (git-ignored, immutable with git SHA in filename)

### Industry IDs
- **US:** `<yfinance_sector>.<static_id>` (frozen at first appearance)
- **CN:** `swL1.<code>` (Shenwan Level 1, point-in-time snapshots)

### Date Handling
- **All dates are exchange-local** (US: NYSE calendar, CN: SSE calendar)
- Logs include both local date and UTC timestamp
- Monthly reconstitution: **last trading day** of each month per region

## Testing Strategy

### Test Types
- **Unit:** `tests/unit/` → test individual functions, no I/O
- **Integration:** `tests/integration/` → test module CLIs, use fixtures for data

### Acceptance Tests (per Sprint 0 spec)
- **Data smoke:** Qlib init succeeds for both regions
- **Universe rules:** N=800, price/volume filters, month-end boundaries (±0 days tolerance)
- **Model 2 CV hygiene:** 63-day embargo enforced, deterministic predictions (seed=42)
- **Optimizer behavior:** tightening turnover τ decreases turnover; lowering vol-target decreases w^T Σ w; industry caps respected within 1e-6
- **Backtest parity:** toy 3-asset P&L within ±1 bp of analytical

### Running Single Tests
```bash
# Single test file
uv run pytest tests/unit/test_logging.py -v

# Single test function
uv run pytest tests/unit/test_logging.py::test_config_hash_deterministic -v

# With coverage for specific module
uv run pytest tests/unit/ --cov=src.utils --cov-report=term-missing
```

## Common Patterns

### Creating a New Module CLI
```python
# src/example/new_module.py
import argparse
import yaml
from src.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--region", choices=["US", "CN"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set up logging with metadata
    logger = setup_logging(
        level="DEBUG" if args.debug else "INFO",
        config=config,
        global_seed=config.get("random_state", 42),
        region=args.region,
    )

    logger.info("Starting new_module", extra={"config_path": args.config})
    # ... implementation ...

if __name__ == "__main__":
    main()
```

### Config Hash Verification
```python
from src.utils.logging import compute_config_hash
import yaml

with open("configs/model2_us.yaml") as f:
    config = yaml.safe_load(f)

config_hash = compute_config_hash(config)  # Deterministic 16-char hex
print(f"Config hash: {config_hash}")
```

## Documentation References

- **Complete specs:** `docs/specs.md` (zero-ambiguity definitions for all sprints)
- **Mathematical foundations:** `docs/theory.md`
- **Project status:** `README.md` (current sprint, roadmap)
- **Change history:** `CHANGELOG.md` (follows Keep a Changelog format)

## Guardrails

### Never Do
- Add new top-level directories without RFC
- Change `pyproject.toml` dependencies without regenerating `uv.lock`
- Run `uv sync` without `--locked` in automated scripts
- Bypass the QP optimizer for weight generation
- Modify locked CLI names/signatures from Sprint 0 spec
- Commit secrets, API keys, or credentials

### Always Do
- Use `make format` before committing
- Run `make test-cov` and verify ≥80% coverage
- Include `random_state=42` in all model configs
- Use `uv run --locked python -m <module>` for all CLIs
- Update `CHANGELOG.md` for any config parameter changes
- Log with `src.utils.logging` for automatic metadata injection
