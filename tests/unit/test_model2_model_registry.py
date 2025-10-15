"""Unit tests for Model 2 model registry.

Per P3C4-001-004: Test ModelRegistry configuration and retrieval.

Tests:
- test_registry_get_ridge: Verify Ridge config matches specs
- test_registry_get_xgboost: Verify XGBoost config matches specs
- test_registry_unknown_model: Verify KeyError for invalid model name
- test_registry_list_models: Verify list_available_models returns correct models
- test_registry_frozen_params_immutable: Verify params deep-copied (mutation protection)
- test_registry_trainer_not_fitted: Verify returned trainer is fresh instance
- test_registry_multiple_calls_independent: Verify multiple calls return independent instances
"""

import numpy as np
import pandas as pd
import pytest

from src.model2.model_registry import (
    RIDGE_PARAMS,
    XGBOOST_PARAMS,
    get_model,
    list_available_models,
)
from src.model2.train import RidgeTrainer, XGBoostTrainer


def test_registry_get_ridge():
    """Verify Ridge configuration matches frozen specs.

    Per P3C4-001-004: alpha=3.0, random_state=42 (NON-NEGOTIABLE)
    """
    result = get_model("ridge")

    # Verify structure
    assert "trainer" in result
    assert "params" in result

    # Verify trainer type
    trainer = result["trainer"]
    assert isinstance(trainer, RidgeTrainer)

    # Verify frozen params
    params = result["params"]
    assert params["alpha"] == 3.0
    assert params["random_state"] == 42

    # Verify trainer params match
    trainer_params = trainer.get_params()
    assert trainer_params == RIDGE_PARAMS


def test_registry_get_xgboost():
    """Verify XGBoost configuration matches frozen specs.

    Per P3C4-001-004: All 6 hyperparameters must match frozen values.
    """
    result = get_model("xgboost")

    # Verify structure
    assert "trainer" in result
    assert "params" in result

    # Verify trainer type
    trainer = result["trainer"]
    assert isinstance(trainer, XGBoostTrainer)

    # Verify all 6 frozen params
    params = result["params"]
    assert params["max_depth"] == 6
    assert params["n_estimators"] == 400
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["random_state"] == 42

    # Verify trainer params match
    trainer_params = trainer.get_params()
    assert trainer_params == XGBOOST_PARAMS


def test_registry_unknown_model():
    """Verify KeyError raised for invalid model name.

    Per P3C4-001-004: Unknown model name should raise KeyError with available models.
    """
    with pytest.raises(KeyError) as exc_info:
        get_model("invalid_model")

    # Verify error message contains available models
    error_msg = str(exc_info.value)
    assert "ridge" in error_msg
    assert "xgboost" in error_msg
    assert "invalid_model" in error_msg


def test_registry_list_models():
    """Verify list_available_models returns correct model names.

    Per P3C4-001-004: Should return ['ridge', 'xgboost'] (sorted).
    """
    models = list_available_models()

    # Verify return type and content
    assert isinstance(models, list)
    assert models == ["ridge", "xgboost"]

    # Verify sorted
    assert models == sorted(models)


def test_registry_frozen_params_immutable():
    """Verify modifying returned params doesn't affect registry.

    Defensive test: ensure registry params are not mutable references.
    """
    result1 = get_model("ridge")
    params1 = result1["params"]

    # Modify returned params
    params1["alpha"] = 999.0
    params1["random_state"] = 999

    # Get model again
    result2 = get_model("ridge")
    params2 = result2["params"]

    # Verify second call returns original frozen values
    assert params2["alpha"] == 3.0
    assert params2["random_state"] == 42


def test_registry_trainer_not_fitted():
    """Verify returned trainer is not yet fitted.

    Per P3C4-001-004: get_model should return fresh trainer instance.
    """
    result = get_model("ridge")
    trainer = result["trainer"]

    # Verify trainer not fitted
    assert trainer._is_fitted is False

    # Verify predict raises error before fit
    X_dummy = pd.DataFrame(np.random.randn(10, 5))
    with pytest.raises(RuntimeError, match="must be fitted"):
        trainer.predict(X_dummy)


def test_registry_multiple_calls_independent():
    """Verify multiple get_model calls return independent trainer instances.

    Ensure no shared state between trainers.
    """
    result1 = get_model("ridge")
    result2 = get_model("ridge")

    trainer1 = result1["trainer"]
    trainer2 = result2["trainer"]

    # Verify different objects
    assert id(trainer1) != id(trainer2)

    # Fit first trainer
    X_train = pd.DataFrame(np.random.randn(100, 5))
    y_train = pd.Series(np.random.randn(100))
    trainer1.fit(X_train, y_train)

    # Verify first trainer is fitted
    assert trainer1._is_fitted is True

    # Verify second trainer still not fitted
    assert trainer2._is_fitted is False
