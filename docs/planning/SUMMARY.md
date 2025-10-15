# P3C4-001 Planning Summary

## Overview
**Ticket**: P3C4-001 - Base Models Training (Model 2)  
**Scope**: Chunk 4 from Phase 3 breakdown  
**Objective**: Train Ridge and XGBoost base models with CPCV, generate OOF predictions

## Deliverables

### 1. Planning Artifacts
- **spec.md** (204 lines): High-level architecture, interfaces, acceptance criteria
- **work_items.json** (12 tickets): Function-level implementation tickets
- **JSON Schemas** (3 files): Stable I/O contracts for outputs

### 2. Work Breakdown (12 Tickets)

#### Foundation (3 tickets)
- **P3C4-001-001**: BaseModelTrainer Abstract Interface
- **P3C4-001-002**: RidgeTrainer Implementation
- **P3C4-001-003**: XGBoostTrainer Implementation

#### Core Pipeline (4 tickets)
- **P3C4-001-004**: ModelRegistry Configuration
- **P3C4-001-005**: OOF Prediction Aggregation
- **P3C4-001-006**: CV Training Loop Orchestrator
- **P3C4-001-007**: Feature Importance Extraction

#### Persistence & Integration (3 tickets)
- **P3C4-001-008**: Model Persistence (Save/Load)
- **P3C4-001-009**: Multi-Horizon Training Wrapper
- **P3C4-001-010**: CV Score Logging and Aggregation

#### Validation (2 tickets)
- **P3C4-001-011**: Determinism Validation Test
- **P3C4-001-012**: Integration with Existing Chunks

### 3. JSON Schema Contracts

All schemas stored in `/home/donaldshen27/projects/donald_trading_model/contracts/`:

1. **OOFPredictions.schema.json**
   - Format: Parquet with MultiIndex (instrument, datetime)
   - Columns: prediction (float), fold_id (int)
   - Used by: All base model trainers

2. **FeatureImportance.schema.json**
   - Format: Parquet with flat index
   - Columns: feature (str), importance_gain (float), importance_weight (float)
   - Used by: XGBoostTrainer only

3. **CVScores.schema.json**
   - Format: JSON log lines
   - Fields: model, horizon, fold_id, metric, value
   - Used by: CV loop orchestrator

## Key Design Decisions (Validated by Codex)

1. **Separate OOF Storage per Model**: Each model-horizon pair writes separate parquet
   - Enables independent diagnostics
   - Prevents leakage
   - Simplifies downstream stacking

2. **Single Orchestrator Module**: Shared CV loop with pluggable model trainers
   - `base_models.py` for trainers and orchestration
   - `model_registry.py` for configuration
   - Reduces code duplication

3. **Structured Feature Importance**: Persisted as Parquet with version/model metadata
   - Enables auditing
   - Supports comparison across runs
   - Top-20 logged for quick inspection

4. **CV Fold Validation**: Enforce min 100 samples per fold
   - Prevents unstable training
   - Fails fast with clear error
   - Logs fold distributions

## Dependencies

### Upstream (Must Exist)
- Chunk 1: `src/model2/labels.py` (build_labels_from_config)
- Chunk 2: `src/model2/features.py` (build_features_from_config)
- Chunk 3: `src/model2/train.py` (PurgedEmbargoedTimeSeriesSplit)
- `src/utils/logging.py` (setup_logging)

### Libraries
- scikit-learn ≥ 1.3 (Ridge)
- xgboost ≥ 1.7 (XGBRegressor)
- joblib (persistence)
- pandas ≥ 2.0, numpy ≥ 1.24

## Acceptance Criteria

### Functional
- All 4 models train (Ridge-21d, Ridge-63d, XGB-21d, XGB-63d)
- OOF predictions cover 100% of training data
- Predictions finite and in [-1000, +1000] bps range
- Deterministic: max_diff < 1e-9 (Ridge), < 1e-6 (XGBoost)

### Performance
- Training time per region: ≤ 2 hours on 8 vCPU / 32 GB RAM
- Memory usage: peak RSS ≤ 24 GB
- Model files: < 500 MB per region

### Quality
- CV scores logged for all 20 fold-model-horizon combos
- Feature importance extracted for XGBoost (top 20 logged)
- No warnings from sklearn/xgboost

## Risk Mitigation

1. **Training Time Risk**: Profile on subset, parallelize folds if needed
2. **Memory Risk**: Monitor RSS, use chunked prediction if OOM
3. **Non-Determinism Risk**: Explicit tests, verify XGBoost seed propagation

## Next Steps

1. Review planning artifacts with team
2. Prioritize work items (suggest sequential: 001→002→003→004→005→006→...)
3. Assign implementer for each ticket
4. Run through skeletoner → implementer → integrator workflow per CLAUDE.md
