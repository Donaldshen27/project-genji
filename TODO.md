# Sprint 0 Implementation Plan

**Sprint 0 Spec Reference:** `docs/specs.md` Section 1 (lines 38-73)

## Overview

Implement complete E2E baseline pipeline: Data → Model 2 → QP Optimizer → Backtest → Report for US & CN regions, with zero RL and zero Model 1 (per specs Section 1).

## 🔗 Key References

- **Specs:** `docs/specs.md` Section 1 (Sprint 0)
- **Theory:** `docs/theory.md`
- **Project Guide:** `CLAUDE.md`
- **Qlib:** https://qlib.readthedocs.io/
- **CVXPY:** https://www.cvxpy.org/

---

## Phase 1: Configuration Foundation

Create 7 YAML config files with authoritative keys:
- `configs/data_{us,cn}.yaml` (ingestion params)
- `configs/model2_{us,cn}.yaml` (CPCV with 63-day embargo, seed=42)
- `configs/optimizer.yaml` (risk/cost model, constraints)
- `configs/backtest_{us,cn}.yaml` (executor, metrics)

---

## Phase 2: Data Pipeline (`src/data/`)

4 modules with CLIs:

1. **`ingest_yf_us.py`** - yfinance download, save to `data/qlib/us_data/`
2. **`ingest_ak_cn.py`** - akshare (Tushare), save to `data/qlib/cn_data/`
3. **`build_universe.py`** - N=800, monthly recon, price ≥ $3.00
4. **`build_qlib_dataset.py`** - convert to Qlib binary format

---

## Phase 3: Model 2 - Stock Alpha (`src/model2/`)

3 modules:

- **`features.py`** - industry-relative, winsorized, PIT-aligned
- **`labels.py`** - forward returns (21d, 63d horizons)
- **`train.py`** - Ridge + XGBoost, CPCV, stacking, isotonic calibration

---

## Phase 4: Optimizer (`src/optimizer/`)

3 modules:

- **`risk_model.py`** - Ledoit-Wolf covariance
- **`cost_model.py`** - L1 + L2 costs
- **`solve_qp.py`** - cvxpy QP with OSQP/Clarabel, industry caps, vol-target

---

## Phase 5: Backtest (`src/backtest/`)

2 modules:

- **`run_backtest.py`** - pyqlib integration, apply weights, compute P&L
- **`report.py`** - HTML with Sharpe/DD/turnover/costs/exposures + git SHA

---

## Phase 6: Acceptance Tests (`tests/`)

5 test suites (all must pass):

1. **`test_data_smoke.py`** - qlib.init() succeeds
2. **`test_universe_rules.py`** - N=800, filters, boundaries
3. **`test_model2_cv_hygiene.py`** - 63-day embargo, seed=42
4. **`test_optimizer_behavior.py`** - monotone responses, caps within 1e-6
5. **`test_backtest_parity.py`** - toy case ±1 bp, full metrics

---

## Phase 7: Makefile Integration

Replace placeholder echo commands with actual module calls for all 12 targets:
- `data_{us,cn}`
- `train_{us,cn}`
- `opt_{us,cn}`
- `backtest_{us,cn}`
- `report_{us,cn}`
- `e2e_{us,cn}`

---

## Phase 8: Final Validation

- Run `make e2e_us` and `make e2e_cn` (both must complete)
- Verify all tests pass: `make test`
- Verify coverage ≥ 80%: `make test-cov`
- Validate CI passes (format, lint, tests)

---

## Estimated Timeline

**4 weeks**

## Non-Negotiables

- Seed=42
- 63-day embargo
- Runtime budgets: data ≤6h, train ≤2h, backtest ≤1h