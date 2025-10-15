# P3C4-001: Base Models Training (Model 2) - Specification

## 1. Scope

Implement training of Ridge and XGBoost base models for Model 2 stock alpha forecasting (Chunk 4 of Phase 3). Generate out-of-fold predictions for two label horizons (21d, 63d) using CPCV splits from Chunk 3.

### 1.1 In Scope
- Ridge regression training (alpha=3.0, random_state=42)
- XGBoost training (max_depth=6, n_estimators=400, eta=0.05, subsample=0.8, colsample=0.8, random_state=42)
- Separate training for 21d and 63d horizons (4 models total: Ridge-21d, Ridge-63d, XGB-21d, XGB-63d)
- OOF predictions using CPCV from Chunk 3
- Per-fold CV score logging
- XGBoost feature importance extraction and logging
- Model persistence to disk

### 1.2 Out of Scope
- Stacking/meta-learning (Chunk 5)
- Isotonic calibration (Chunk 5)
- Multi-horizon combination (Chunk 5)
- Style neutralization (Chunk 6)
- CLI integration (Chunk 7)
- Acceptance tests (Chunk 8)

### 1.3 Non-Goals
- Hyperparameter tuning (parameters frozen per specs)
- Alternative model architectures
- Online/incremental learning
- GPU acceleration

## 2. Architecture

### 2.1 Module Structure
```
src/model2/
├── train.py              # Main orchestrator (Chunks 3-7)
├── base_models.py        # NEW: Base model training logic
└── model_registry.py     # NEW: Model configuration registry
```

### 2.2 Core Components

**BaseModelTrainer (Abstract)**
- Interface for training individual models
- Methods: `fit(X_train, y_train)`, `predict(X_test)`, `get_feature_importance()`
- Implementations: `RidgeTrainer`, `XGBoostTrainer`

**ModelRegistry**
- Central configuration for all base models
- Maps model names to trainer classes and hyperparameters
- Ensures reproducibility with seed propagation

**TrainingOrchestrator**
- Coordinates CV loop, data preparation, model training
- Aggregates OOF predictions across folds
- Manages logging and persistence

### 2.3 Data Flow
```
Input: Features (from Chunk 2) + Labels (from Chunk 1) + CPCV splits (from Chunk 3)
  ↓
For each horizon (21d, 63d):
  For each model (Ridge, XGBoost):
    For each CV fold (5 folds):
      - Train on train_idx
      - Predict on test_idx → OOF predictions
      - Log fold score
    - Aggregate OOF predictions
    - Train final model on all data
    - Save model + OOF predictions
```

## 3. Interfaces

### 3.1 Input Contracts
- **Features**: DataFrame with MultiIndex (instrument, datetime), shape (N_samples, N_features)
- **Labels**: DataFrame with MultiIndex (instrument, datetime), columns [label_21d, label_63d]
- **CV Splitter**: PurgedEmbargoedTimeSeriesSplit instance from Chunk 3

### 3.2 Output Contracts

**OOF Predictions** (per model, per horizon)
- Location: `data/model2/{region}/oof/{model}_{horizon}.parquet`
- Schema:
  - Index: MultiIndex (instrument, datetime)
  - Columns: `[prediction, fold_id]`
  - Type: float32 for predictions, int8 for fold_id

**Trained Models**
- Location: `models/model2_{region}/{model}_{horizon}.pkl`
- Format: joblib pickle
- Contents: scikit-learn/xgboost model object

**Feature Importance** (XGBoost only)
- Location: `models/model2_{region}/feature_importance_{horizon}.parquet`
- Schema: `[feature: str, importance_gain: float32, importance_weight: float32]`

**CV Scores Log**
- Location: Logged via src.utils.logging
- Format: JSON lines with keys: `model, horizon, fold_id, metric, value`

### 3.3 JSON Schema References
Store output schemas in `/home/donaldshen27/projects/donald_trading_model/contracts/`:
- `OOFPredictions.schema.json` (OOF predictions format)
- `FeatureImportance.schema.json` (feature importance format)
- `CVScores.schema.json` (CV scores logging format)

## 4. Acceptance Criteria

### 4.1 Functional Requirements
1. All 4 models (Ridge-21d, Ridge-63d, XGB-21d, XGB-63d) train successfully
2. OOF predictions cover 100% of training data (no gaps after CV aggregation)
3. Predictions are finite (no NaN, no Inf)
4. Predictions within reasonable range: [-1000, +1000] bps
5. Deterministic: Two runs with seed=42 produce identical OOF predictions (max diff < 1e-9)

### 4.2 Performance Requirements
1. Training time per region: ≤ 2 hours on 8 vCPU / 32 GB RAM
2. Memory usage: peak RSS ≤ 24 GB
3. Model files: total size < 500 MB per region

### 4.3 Quality Requirements
1. CV scores logged for all folds (5 folds × 2 horizons × 2 models = 20 entries)
2. Feature importance extracted for XGBoost (top 20 features logged)
3. No warnings from sklearn/xgboost (e.g., convergence issues)

## 5. Edge Cases & Error Handling

### 5.1 Data Quality Issues
- **Empty fold**: Raise ValueError with fold info
- **Insufficient samples**: Validate min samples per fold (≥ 100) before training
- **Label NaN in fold**: Drop samples with NaN labels, log count

### 5.2 Model Training Issues
- **Ridge singular matrix**: Use Ridge solver with regularization (guaranteed to work)
- **XGBoost early stopping**: Disable early stopping (n_estimators fixed)
- **OOM during XGBoost**: Reduce tree_method to 'hist' if default fails

### 5.3 Prediction Issues
- **Prediction NaN**: Raise ValueError, do not silently drop
- **Prediction Inf**: Raise ValueError with model and fold info
- **Outlier predictions**: Log warning if >1% of predictions exceed ±500 bps

## 6. Testing Strategy

### 6.1 Unit Tests (tickets/work_items.json)
- `test_ridge_trainer`: Verify Ridge training and prediction
- `test_xgboost_trainer`: Verify XGBoost training and prediction with feature importance
- `test_oof_aggregation`: Verify OOF predictions cover all samples
- `test_determinism`: Verify seed=42 produces identical results

### 6.2 Integration Tests
- `test_full_cv_loop`: End-to-end CV training with synthetic data
- `test_output_schemas`: Validate all outputs match JSON schemas
- `test_model_persistence`: Save and load models, verify predictions match

### 6.3 Smoke Tests
- Train on small subset (100 samples, 2 folds) to verify pipeline

## 7. Dependencies

### 7.1 Upstream (Must Exist)
- Chunk 1: `src/model2/labels.py` (build_labels_from_config)
- Chunk 2: `src/model2/features.py` (build_features_from_config)
- Chunk 3: `src/model2/train.py` (PurgedEmbargoedTimeSeriesSplit, create_cv_from_config)
- `src/utils/logging.py` (setup_logging, get_logger)

### 7.2 External Libraries
- scikit-learn ≥ 1.3 (Ridge, model_selection)
- xgboost ≥ 1.7 (XGBRegressor)
- joblib (model persistence)
- pandas ≥ 2.0
- numpy ≥ 1.24

## 8. Risks & Mitigations

### 8.1 Risk: Training Time Exceeds Budget
- **Likelihood**: Medium (XGBoost with 400 trees on large dataset)
- **Impact**: High (blocks E2E pipeline)
- **Mitigation**: Profile on subset, consider tree_method='hist', parallelize folds

### 8.2 Risk: Memory Overflow
- **Likelihood**: Low (Ridge is memory-efficient, XGBoost scales well)
- **Impact**: High (OOM crashes)
- **Mitigation**: Monitor RSS during development, use chunked prediction if needed

### 8.3 Risk: Non-Determinism
- **Likelihood**: Low (all seeds fixed)
- **Impact**: High (reproducibility violated)
- **Mitigation**: Test determinism explicitly, verify XGBoost seed propagation

## 9. Open Questions (For Implementation)
1. Should OOF predictions include confidence intervals?
   → Deferred to Chunk 5 (variance estimation)
2. Should we log training curves for XGBoost?
   → Nice-to-have, not required for Chunk 4
3. Should Ridge use cross-validation to select alpha?
   → No, alpha=3.0 is frozen per specs

## 10. Success Metrics
- All 4 models saved to `models/model2_{region}/`
- All 4 OOF prediction files exist and pass schema validation
- Feature importance file exists for XGBoost
- All CV scores logged (20 entries in logs)
- Determinism test passes (identical predictions on re-run)
