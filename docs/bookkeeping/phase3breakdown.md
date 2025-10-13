# Phase 3: Model 2 Implementation Breakdown

**CURRENT WORKING CHUNK: 1**

## Overview

Implement complete E2E baseline pipeline: Data → Model 2 → QP Optimizer → Backtest → Report for US & CN regions, with zero RL and zero Model 1 (per specs Section 1).

## 🔗 Key References

- **Specs:** `docs/specs.md` Section 1 (Sprint 0)
- **Theory:** `docs/theory.md`
- **Project Guide:** `CLAUDE.md`
- **Qlib:** https://qlib.readthedocs.io/
- **CVXPY:** https://www.cvxpy.org/

---

## Chunk 1: Labels Module (`src/model2/labels.py`)
- [ ] Create industry-relative forward returns function
- [ ] Input: Qlib dataset, horizons=[21, 63]
- [ ] Compute: `y_{i,t}^{(k)} = sum(r_{i,t+1:t+k}) - sum(R_ind_{i,t+1:t+k})`
- [ ] Industry proxies: synthetic equal-weighted returns from universe
- [ ] Output schema: `[date, instrument, label_21d, label_63d]`
- [ ] No look-ahead: labels use only future returns, aligned to prediction date t

## Chunk 2: Features Module - Technical Only (`src/model2/features.py`)
- [ ] Build technical features from Qlib price/volume data
- [ ] Features: momentum (12-1, 1m reversal), MA gaps (20/50/200), RSI, realized vol, drawdown stats
- [ ] Microstructure: volume ratios, turnover, Amihud illiquidity
- [ ] Preprocessing: winsorize [1%, 99%], then industry z-score at each date
- [ ] Output schema: `[date, instrument, feature_1, ..., feature_N]`
- [ ] Handle missing: forward-fill (limit=5), then dropna

## Chunk 3: CPCV Implementation (`src/model2/train.py` - Part 1)
- [ ] TimeSeriesSplit with n_splits=5 (expanding window)
- [ ] Purge logic: remove samples from train if label window overlaps test
- [ ] Embargo: 63-day gap between train end and test start (NON-NEGOTIABLE)
- [ ] Return: list of (train_idx, test_idx) tuples
- [ ] Validate: no overlap within embargo window

## Chunk 4: Base Models Training (`src/model2/train.py` - Part 2)
- [ ] Ridge: alpha=3.0, random_state=42
- [ ] XGBoost: max_depth=6, n_estimators=400, eta=0.05, subsample=0.8, colsample=0.8, random_state=42
- [ ] Train separately for 21d and 63d labels
- [ ] Generate OOF predictions using CPCV folds
- [ ] Log: CV scores per fold, feature importance (XGB)

## Chunk 5: Stacking & Calibration (`src/model2/train.py` - Part 3)
- [ ] Meta-learner: Ridge(alpha=3.0) on OOF predictions
- [ ] Isotonic calibration per horizon: fit on OOF, map scores → bp alpha
- [ ] Variance estimation: std of residuals per horizon
- [ ] Multi-horizon combination: `α_final = Σ(ω_k * α_k)` where `ω_k ∝ 1/Var(α_k)`

## Chunk 6: Neutralization & Output (`src/model2/train.py` - Part 4)
- [ ] If config.neutralization.style_neutral==true: regress α on beta/size/momentum, take residuals
- [ ] Save models: `models/model2_{region}/[ridge|xgb|meta|calibrator]_{21d|63d}.pkl`
- [ ] Save predictions: `data/model2/{region}/predictions.parquet`
- [ ] Schema: `[date, instrument, alpha_21d, alpha_63d, alpha_combined, var_21d, var_63d]`
- [ ] Log: config_hash, git_sha, CV scores, runtime

## Chunk 7: CLI & Integration (`src/model2/train.py` - Part 5)
- [ ] Argparse: --config (required), --debug (optional)
- [ ] Load config from YAML, normalize dates
- [ ] Setup logging with metadata (config_hash, seed=42, region)
- [ ] Main flow: load data → build labels → build features → train → save
- [ ] Update Makefile: replace echo with actual command in train_us/train_cn

## Chunk 8: Acceptance Test (`tests/test_model2_cv_hygiene.py`)
- [ ] Test: 63-day embargo enforced (check all fold pairs)
- [ ] Test: deterministic predictions under seed=42 (run twice, compare)
- [ ] Test: output schema matches specs
- [ ] Test: predictions finite and in [-1000, +1000] bps range
- [ ] Test: no future data leakage (manual audit of feature construction)

---

## Dependencies
- Qlib datasets available (Phase 2 ✅)
- Universe files: `data/universe_{us,cn}.parquet` ✅
- Logging utility: `src.utils.logging` ✅

## Runtime Target
≤ 2 hours per region (enforced in specs)

## Success Criteria
- All 8 chunks completed
- `make train_us` and `make train_cn` execute successfully
- Embargo hygiene test passes
- Predictions saved to `data/model2/{region}/predictions.parquet`
