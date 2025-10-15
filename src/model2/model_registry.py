"""Model Registry for Base Model Configurations.

Per P3C4-001-004: Central registry mapping model names to trainer classes and hyperparameters.

Provides:
- get_model(model_name) -> dict with trainer and params
- list_available_models() -> list of model names
- Frozen hyperparameters per specs Section 1

Raises:
- KeyError: Unknown model name
- ValueError: Invalid hyperparameters
"""

import copy
import logging
import math
from typing import Any, Literal

from src.model2.train import RidgeTrainer, XGBoostTrainer

logger = logging.getLogger(__name__)

# Type alias for valid model names
ModelName = Literal["ridge", "xgboost"]


# ============================================================================
# Frozen Hyperparameter Configurations
# ============================================================================
# Per P3C4-001-002 and P3C4-001-003: All hyperparameters are NON-NEGOTIABLE

RIDGE_PARAMS: dict[str, Any] = {
    "alpha": 3.0,
    "random_state": 42,
}

XGBOOST_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "n_estimators": 400,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


# ============================================================================
# Model Registry
# ============================================================================

_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "ridge": {
        "trainer_class": RidgeTrainer,
        "params": RIDGE_PARAMS,
    },
    "xgboost": {
        "trainer_class": XGBoostTrainer,
        "params": XGBOOST_PARAMS,
    },
}


def get_model(model_name: str) -> dict[str, Any]:
    """Get trainer instance and hyperparameters for a registered model.

    Per P3C4-001-004: Returns a dict with 'trainer' (BaseModelTrainer instance)
    and 'params' (frozen hyperparameters).

    Args:
        model_name: Model identifier ("ridge" or "xgboost")

    Returns:
        Dictionary with keys:
        - 'trainer': BaseModelTrainer instance (not yet fitted)
        - 'params': dict of frozen hyperparameters

    Raises:
        KeyError: If model_name is not registered
        ValueError: If hyperparameters are invalid (should never occur with frozen params)

    Example:
        >>> result = get_model("ridge")
        >>> trainer = result["trainer"]  # RidgeTrainer instance
        >>> params = result["params"]    # {"alpha": 3.0, "random_state": 42}
        >>> trainer.fit(X_train, y_train)
    """
    if model_name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model name '{model_name}'. Available models: {available}")

    registry_entry = _MODEL_REGISTRY[model_name]
    trainer_class = registry_entry["trainer_class"]
    frozen_params = registry_entry["params"]

    # Deep copy params to prevent mutation
    params_copy = copy.deepcopy(frozen_params)

    # Validate params match frozen values (defensive check)
    _validate_hyperparameters(model_name, params_copy)

    # Instantiate trainer with frozen params
    # Trainer __init__ will validate params match frozen values
    trainer = trainer_class(**params_copy)

    logger.debug(
        "Retrieved model '%s': trainer=%s, params=%s",
        model_name,
        trainer_class.__name__,
        params_copy,
    )

    return {
        "trainer": trainer,
        "params": params_copy,
    }


def list_available_models() -> list[str]:
    """List all registered model names.

    Per P3C4-001-004: Returns sorted list of available model identifiers.

    Returns:
        List of model names, e.g., ["ridge", "xgboost"]

    Example:
        >>> models = list_available_models()
        >>> assert "ridge" in models
        >>> assert "xgboost" in models
    """
    return sorted(_MODEL_REGISTRY.keys())


def _validate_hyperparameters(model_name: str, params: dict[str, Any]) -> None:
    """Validate hyperparameters against frozen values.

    Defensive check to ensure no accidental parameter drift.

    Args:
        model_name: Model identifier
        params: Hyperparameters to validate

    Raises:
        ValueError: If any parameter differs from frozen value

    Note:
        This function is internal and called by get_model() as a sanity check.
        With frozen params, this should never raise.
    """
    if model_name not in _MODEL_REGISTRY:
        return  # Let get_model handle unknown models

    frozen_params = _MODEL_REGISTRY[model_name]["params"]

    # Check all frozen params are present
    missing_params = set(frozen_params.keys()) - set(params.keys())
    if missing_params:
        raise ValueError(
            f"Model '{model_name}' is missing required hyperparameters: {sorted(missing_params)}"
        )

    # Check all params match frozen values
    for param_name, frozen_value in frozen_params.items():
        actual_value = params.get(param_name)

        # Compare with appropriate tolerance for floats
        if isinstance(frozen_value, float):
            if not math.isclose(actual_value, frozen_value, abs_tol=1e-12):
                raise ValueError(
                    f"Model '{model_name}' hyperparameter '{param_name}' must be {frozen_value} "
                    f"(got {actual_value}). Hyperparameters are frozen per specs Section 1."
                )
        else:
            if actual_value != frozen_value:
                raise ValueError(
                    f"Model '{model_name}' hyperparameter '{param_name}' must be {frozen_value} "
                    f"(got {actual_value}). Hyperparameters are frozen per specs Section 1."
                )

    logger.debug(
        "Validated hyperparameters for model '%s' against frozen values.",
        model_name,
    )
