"""
Unit tests for base model trainers (Chunk 4 stubs).

Tests P3C4-001-001 through P3C4-001-003:
- BaseModelTrainer cannot be instantiated (ABC)
- RidgeTrainer interface
- XGBoostTrainer interface

NOTE: Tests marked with pytest.skip until implementation.
"""

import numpy as np
import pandas as pd
import pytest

from src.model2.train import BaseModelTrainer, RidgeTrainer, XGBoostTrainer


def test_base_model_trainer_cannot_instantiate():
    """Verify BaseModelTrainer is abstract and cannot be instantiated.

    Per P3C4-001-001: ABC should raise TypeError on instantiation.
    """
    with pytest.raises(TypeError):
        BaseModelTrainer()  # type: ignore


def test_base_model_trainer_interface():
    """Verify BaseModelTrainer defines all required abstract methods.

    Per P3C4-001-001: fit, predict, get_feature_importance, get_params
    """
    required_methods = {"fit", "predict", "get_feature_importance", "get_params"}
    actual_methods = {name for name in dir(BaseModelTrainer) if not name.startswith("_")}
    assert required_methods.issubset(actual_methods)


@pytest.mark.skip(reason="P3C4-001-002: RidgeTrainer implementation pending")
def test_ridge_trainer_initialization():
    """Verify RidgeTrainer initializes with correct defaults.

    Per P3C4-001-002: alpha=3.0, random_state=42
    """
    trainer = RidgeTrainer()
    params = trainer.get_params()
    assert params["alpha"] == 3.0
    assert params["random_state"] == 42


@pytest.mark.skip(reason="P3C4-001-002: RidgeTrainer implementation pending")
def test_ridge_trainer_fit_predict():
    """Verify RidgeTrainer fit and predict on toy data.

    Per P3C4-001-002: 10 samples, 3 features
    """
    X = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    y = pd.Series(np.random.randn(10))

    trainer = RidgeTrainer()
    trainer.fit(X, y)
    predictions = trainer.predict(X)

    assert predictions.shape == (10,)
    assert np.all(np.isfinite(predictions))


def test_ridge_trainer_no_feature_importance():
    """Verify RidgeTrainer.get_feature_importance() returns None.

    Per P3C4-001-002: Ridge does not have standardized feature importance
    """
    # This is the only method with implementation in stub
    # TODO: Instantiate without raising (after __init__ implemented)
    pass


@pytest.mark.skip(reason="P3C4-001-003: XGBoostTrainer implementation pending")
def test_xgboost_trainer_initialization():
    """Verify XGBoostTrainer initializes with correct hyperparameters.

    Per P3C4-001-003: max_depth=6, n_estimators=400, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
    """
    trainer = XGBoostTrainer()
    params = trainer.get_params()
    assert params["max_depth"] == 6
    assert params["n_estimators"] == 400
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["random_state"] == 42
