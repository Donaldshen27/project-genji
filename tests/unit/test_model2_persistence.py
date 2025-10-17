"""Unit tests for Model 2 persistence utilities."""

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.model2.base_models import RidgeTrainer, XGBoostTrainer
from src.model2 import persistence as persistence_module
from src.model2.persistence import (
    load_oof_predictions,
    load_trained_model,
    save_oof_predictions,
    save_trained_model,
)


def _build_multiindex_oof_df() -> pd.DataFrame:
    instruments = ["AAPL", "GOOGL"]
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    index = pd.MultiIndex.from_product(
        [instruments, dates], names=["instrument", "datetime"]
    )
    predictions = np.linspace(-0.5, 0.5, len(index))
    folds = np.tile(np.arange(5, dtype=np.int8), len(instruments))
    df = pd.DataFrame({"prediction": predictions, "fold_id": folds}, index=index)
    return df


# ---------------------------------------------------------------------------
# Model persistence tests
# ---------------------------------------------------------------------------


def test_save_and_load_ridge_model_round_trip(tmp_path: Path) -> None:
    X = pd.DataFrame(np.random.randn(128, 6), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.random.randn(128))

    trainer = RidgeTrainer()
    trainer.fit(X, y)

    model_path = tmp_path / "models" / "ridge_21d.pkl"
    saved_metadata = save_trained_model(trainer, model_path)

    assert model_path.exists()
    loaded_model, loaded_metadata = load_trained_model(model_path, expected_type=RidgeTrainer)

    X_eval = X.iloc[:10]
    np.testing.assert_allclose(trainer.predict(X_eval), loaded_model.predict(X_eval))
    assert saved_metadata.model_hash == loaded_metadata.model_hash
    assert loaded_metadata.class_path.endswith("RidgeTrainer")


def test_save_and_load_xgboost_model_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"x{i}" for i in range(8)])
    y = pd.Series(rng.normal(size=200))

    trainer = XGBoostTrainer()
    trainer.fit(X, y)

    model_path = tmp_path / "models" / "xgb_63d.pkl"
    saved_metadata = save_trained_model(trainer, model_path)

    loaded_model, loaded_metadata = load_trained_model(model_path, expected_type=XGBoostTrainer)

    X_eval = X.iloc[:20]
    np.testing.assert_allclose(trainer.predict(X_eval), loaded_model.predict(X_eval), atol=1e-9)
    assert saved_metadata.model_hash == loaded_metadata.model_hash
    assert loaded_metadata.class_path.endswith("XGBoostTrainer")


def test_save_model_creates_parent_dir_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    X = pd.DataFrame(np.random.randn(32, 4), columns=[f"c{i}" for i in range(4)])
    y = pd.Series(np.random.randn(32))
    trainer = RidgeTrainer().fit(X, y)

    model_path = tmp_path / "nested" / "ridge.pkl"
    save_trained_model(trainer, model_path)
    assert model_path.exists()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.model2.persistence"):
        save_trained_model(trainer, model_path)
    assert "Overwriting existing artifact" in caplog.text


def test_load_trained_model_type_mismatch(tmp_path: Path) -> None:
    X = pd.DataFrame(np.random.randn(16, 3), columns=[f"f{i}" for i in range(3)])
    y = pd.Series(np.random.randn(16))
    trainer = RidgeTrainer().fit(X, y)

    model_path = tmp_path / "model.pkl"
    save_trained_model(trainer, model_path)

    with pytest.raises(TypeError, match="Expected model type XGBoostTrainer"):
        load_trained_model(model_path, expected_type=XGBoostTrainer)


def test_load_trained_model_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pkl"
    with pytest.raises(FileNotFoundError):
        load_trained_model(missing)


# ---------------------------------------------------------------------------
# OOF persistence tests
# ---------------------------------------------------------------------------


def test_save_and_load_oof_predictions_multiindex(tmp_path: Path) -> None:
    oof_df = _build_multiindex_oof_df()
    parquet_path = tmp_path / "oof" / "ridge_21d.parquet"

    saved_metadata = save_oof_predictions(oof_df, parquet_path)
    assert parquet_path.exists()

    loaded_df, loaded_metadata = load_oof_predictions(parquet_path)

    pd.testing.assert_frame_equal(loaded_df, oof_df.astype({"prediction": np.float32, "fold_id": np.int8}))
    assert saved_metadata.data_hash == loaded_metadata.data_hash
    assert loaded_df.index.names == ("instrument", "datetime")
    assert loaded_df["prediction"].dtype == np.float32
    assert loaded_df["fold_id"].dtype == np.int8


def test_save_oof_predictions_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"prediction": [0.1, 0.2]})
    parquet_path = tmp_path / "oof.parquet"

    with pytest.raises(ValueError, match="(?i)missing required columns"):
        save_oof_predictions(df, parquet_path)


def test_save_oof_predictions_warns_on_overwrite(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    df = _build_multiindex_oof_df()
    parquet_path = tmp_path / "oof.parquet"
    save_oof_predictions(df, parquet_path)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.model2.persistence"):
        save_oof_predictions(df, parquet_path)

    assert "Overwriting existing artifact" in caplog.text


def test_load_oof_predictions_hash_mismatch(tmp_path: Path) -> None:
    df = _build_multiindex_oof_df()
    parquet_path = tmp_path / "oof.parquet"
    save_oof_predictions(df, parquet_path)

    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata_key = persistence_module.OOF_METADATA_KEY
    corrupted = json.loads(metadata[metadata_key].decode("utf-8"))
    corrupted["data_hash"] = "corrupted"
    metadata[metadata_key] = json.dumps(corrupted).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)

    with pytest.raises(ValueError, match="OOF predictions hash mismatch"):
        load_oof_predictions(parquet_path)


def test_save_and_load_empty_oof_predictions(tmp_path: Path) -> None:
    index = pd.MultiIndex.from_arrays([[], []], names=["instrument", "datetime"])
    df = pd.DataFrame({"prediction": pd.Series(dtype=np.float32), "fold_id": pd.Series(dtype=np.int8)}, index=index)
    parquet_path = tmp_path / "empty.parquet"

    save_oof_predictions(df, parquet_path)
    loaded_df, _ = load_oof_predictions(parquet_path)

    assert loaded_df.empty
    assert isinstance(loaded_df.index, pd.MultiIndex)
    assert loaded_df["prediction"].dtype == np.float32
    assert loaded_df["fold_id"].dtype == np.int8
