"""Persistence utilities for Model 2 base models and OOF predictions.

Provides helpers to store and restore trained :class:`BaseModelTrainer`
instances (Ridge, XGBoost) and their out-of-fold (OOF) predictions with
robust error handling. Ensures round-trip integrity by computing stable
hashes of serialized artifacts and embedding them into the saved files.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Generic, TypeVar

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.util import hash_pandas_object

from .base_models import BaseModelTrainer

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Attach a NullHandler so logging uses the normal handler chain (captured by tests)
    logger.addHandler(logging.NullHandler())
logger.propagate = True

# Constants -----------------------------------------------------------------

MODEL_ARTIFACT_VERSION = "1.0"
OOF_ARTIFACT_VERSION = "1.0"
OOF_METADATA_KEY = b"model2_oof_metadata"

T = TypeVar("T", bound=BaseModelTrainer)


# Dataclasses ---------------------------------------------------------------

@dataclass(frozen=True)
class ModelArtifactMetadata(Generic[T]):
    """Metadata stored alongside a serialized model artifact."""

    artifact_version: str
    class_path: str
    created_at_utc: str
    library_versions: dict[str, str]
    model_params: dict[str, Any] | None
    model_hash: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "class_path": self.class_path,
            "created_at_utc": self.created_at_utc,
            "library_versions": self.library_versions,
            "model_params": self.model_params,
            "model_hash": self.model_hash,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelArtifactMetadata":
        default_versions: dict[str, str] = {}
        return cls(
            artifact_version=data.get("artifact_version", MODEL_ARTIFACT_VERSION),
            class_path=data.get("class_path", ""),
            created_at_utc=data.get("created_at_utc", ""),
            library_versions=data.get("library_versions", default_versions),
            model_params=data.get("model_params"),
            model_hash=data.get("model_hash", ""),
            extra=data.get("extra", {}),
        )


@dataclass(frozen=True)
class OOFArtifactMetadata:
    """Metadata embedded into the OOF parquet file."""

    artifact_version: str
    created_at_utc: str
    num_rows: int
    index_names: tuple[str | None, ...]
    column_dtypes: dict[str, str]
    data_hash: str

    def to_json_bytes(self) -> bytes:
        payload = {
            "artifact_version": self.artifact_version,
            "created_at_utc": self.created_at_utc,
            "num_rows": self.num_rows,
            "index_names": list(self.index_names),
            "column_dtypes": self.column_dtypes,
            "data_hash": self.data_hash,
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "OOFArtifactMetadata":
        data = json.loads(payload.decode("utf-8"))
        return cls(
            artifact_version=data.get("artifact_version", OOF_ARTIFACT_VERSION),
            created_at_utc=data.get("created_at_utc", ""),
            num_rows=int(data.get("num_rows", 0)),
            index_names=tuple(data.get("index_names", [])),
            column_dtypes=data.get("column_dtypes", {}),
            data_hash=data.get("data_hash", ""),
        )


# Helpers -------------------------------------------------------------------

def _ensure_parent_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - propagate with context
        raise OSError(f"Failed to create parent directory for {path}: {exc}") from exc


def _warn_on_overwrite(path: Path) -> None:
    if path.exists():
        logger.warning("Overwriting existing artifact at %s", path)


def _atomic_joblib_dump(obj: Any, path: Path, *, compress: int = 3) -> None:
    tmp_file = tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=path.stem, suffix=".tmp", delete=False
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    try:
        joblib.dump(obj, tmp_path, compress=compress)
        os.replace(tmp_path, path)
    except Exception as exc:  # pragma: no cover - re-raise with cleanup
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise OSError(f"Failed to write joblib artifact to {path}: {exc}") from exc


def _atomic_parquet_write(table: pa.Table, path: Path) -> None:
    tmp_file = tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=path.stem, suffix=".tmp", delete=False
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:  # pragma: no cover - re-raise with cleanup
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise OSError(f"Failed to write parquet artifact to {path}: {exc}") from exc


def _current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_class_path(model: BaseModelTrainer) -> str:
    return f"{model.__class__.__module__}.{model.__class__.__name__}"


def _collect_library_versions() -> dict[str, str]:
    return {
        "joblib": getattr(joblib, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
        "pandas": getattr(pd, "__version__", "unknown"),
        "pyarrow": getattr(pa, "__version__", "unknown"),
    }


def _build_model_metadata(model: BaseModelTrainer) -> ModelArtifactMetadata:
    metadata = ModelArtifactMetadata(
        artifact_version=MODEL_ARTIFACT_VERSION,
        class_path=_model_class_path(model),
        created_at_utc=_current_utc_timestamp(),
        library_versions=_collect_library_versions(),
        model_params=model.get_params(),
        model_hash=joblib.hash(model),
        extra={},
    )
    return metadata


def _validate_model_is_fitted(model: BaseModelTrainer) -> None:
    if hasattr(model, "_is_fitted") and not getattr(model, "_is_fitted"):
        raise ValueError("Cannot persist an unfitted model trainer.")


def _prepare_oof_frame(oof_df: pd.DataFrame, *, copy_frame: bool = True) -> pd.DataFrame:
    if not isinstance(oof_df, pd.DataFrame):
        raise TypeError("oof_df must be a pandas DataFrame.")

    required_columns = {"prediction", "fold_id"}
    missing = required_columns - set(oof_df.columns)
    if missing:
        raise ValueError(
            f"OOF predictions missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(required_columns)}"
        )

    frame = oof_df.copy(deep=True) if copy_frame else oof_df

    # Enforce dtypes for deterministic serialization
    try:
        frame["prediction"] = frame["prediction"].astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("OOF 'prediction' column must be numeric.") from exc

    try:
        frame["fold_id"] = frame["fold_id"].astype(np.int8)
    except (TypeError, ValueError) as exc:
        raise ValueError("OOF 'fold_id' column must be integer coercible.") from exc

    if not np.isfinite(frame["prediction"].to_numpy()).all():
        raise ValueError("OOF predictions contain NaN or infinite values.")

    if isinstance(frame.index, pd.MultiIndex):
        index_names = tuple(frame.index.names)
        if any(name is None for name in index_names):
            logger.warning(
                "OOF MultiIndex has unnamed levels; consider setting explicit names for clarity."
            )
    else:
        logger.warning(
            "OOF predictions do not use a MultiIndex; downstream joins may rely on MultiIndex indices."
        )

    return frame


def _fingerprint_dataframe(df: pd.DataFrame) -> str:
    hashed = hash_pandas_object(df, index=True, categorize=True)
    return sha256(hashed.values.tobytes()).hexdigest()


def _build_oof_metadata(oof_df: pd.DataFrame) -> OOFArtifactMetadata:
    if isinstance(oof_df.index, pd.MultiIndex):
        index_names = tuple(oof_df.index.names)
    else:
        index_names = (oof_df.index.name,)
    metadata = OOFArtifactMetadata(
        artifact_version=OOF_ARTIFACT_VERSION,
        created_at_utc=_current_utc_timestamp(),
        num_rows=int(len(oof_df)),
        index_names=index_names,
        column_dtypes={col: str(oof_df[col].dtype) for col in oof_df.columns},
        data_hash=_fingerprint_dataframe(oof_df),
    )
    return metadata


def _prepare_oof_table(oof_df: pd.DataFrame) -> pa.Table:
    try:
        table = pa.Table.from_pandas(oof_df, preserve_index=True)
    except Exception as exc:  # pragma: no cover - propagate with context
        raise ValueError(f"Failed to convert OOF predictions to pyarrow Table: {exc}") from exc
    return table


# Public API ----------------------------------------------------------------

def save_trained_model(
    model: BaseModelTrainer,
    output_path: str | Path,
    *,
    compress: int = 3,
    extra_metadata: dict[str, Any] | None = None,
) -> ModelArtifactMetadata:
    """
    Persist a fitted :class:`BaseModelTrainer` instance to disk via joblib.

    Args:
        model: Trained model trainer (RidgeTrainer, XGBoostTrainer, etc.).
        output_path: Destination ``.pkl`` file path.
        compress: joblib compression level (default 3).
        extra_metadata: Optional dictionary merged into metadata["extra"].

    Returns:
        ``ModelArtifactMetadata`` describing the saved artifact.

    Raises:
        ValueError: If the model is unfitted or path suffix is incorrect.
        OSError: If disk operations fail (permissions, disk full, etc.).
    """
    output = Path(output_path)
    if output.suffix != ".pkl":
        raise ValueError(f"Model artifact path must end with '.pkl' (got {output}).")

    _validate_model_is_fitted(model)
    _ensure_parent_dir(output)
    _warn_on_overwrite(output)

    metadata = _build_model_metadata(model)
    if extra_metadata:
        metadata.extra.update(extra_metadata)

    artifact_payload = {
        "model": model,
        "metadata": metadata.to_dict(),
    }

    _atomic_joblib_dump(artifact_payload, output, compress=compress)

    logger.info(
        "Saved model artifact for %s to %s (hash=%s)",
        metadata.class_path,
        output,
        metadata.model_hash,
    )
    return metadata


def load_trained_model(
    input_path: str | Path,
    expected_type: type[T] | None = None,
    *,
    verify_hash: bool = True,
) -> tuple[T, ModelArtifactMetadata]:
    """
    Load a persisted :class:`BaseModelTrainer` instance from disk.

    Args:
        input_path: Path to the ``.pkl`` artifact.
        expected_type: Optional subclass of :class:`BaseModelTrainer` to enforce.
        verify_hash: Whether to compare stored hash with recomputed value.

    Returns:
        Tuple of ``(model, metadata)``.

    Raises:
        FileNotFoundError: If the artifact path does not exist.
        OSError: If joblib fails to deserialize the payload.
        TypeError: If the loaded object is not the expected trainer type.
        ValueError: Hash mismatch indicating potential corruption.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {input_path}.")

    try:
        payload = joblib.load(input_path)
    except Exception as exc:  # pragma: no cover - propagate with context
        raise OSError(f"Failed to load model artifact from {input_path}: {exc}") from exc

    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        metadata_dict = payload.get("metadata", {})
        metadata = ModelArtifactMetadata.from_dict(metadata_dict)
    else:
        model = payload
        metadata = _build_model_metadata(model)
        logger.warning(
            "Model artifact at %s is missing metadata; generated metadata on load.", input_path
        )

    if not isinstance(model, BaseModelTrainer):
        raise TypeError(
            f"Loaded object from {input_path} is not a BaseModelTrainer (got {type(model)})."
        )

    if expected_type and not isinstance(model, expected_type):
        raise TypeError(
            f"Expected model type {expected_type.__name__}, "
            f"but loaded {type(model).__name__} from {input_path}."
        )

    if verify_hash:
        current_hash = joblib.hash(model)
        if metadata.model_hash and metadata.model_hash != current_hash:
            raise ValueError(
                f"Model hash mismatch for {input_path}. "
                f"Expected {metadata.model_hash}, observed {current_hash}."
            )

    logger.info(
        "Loaded model artifact from %s (class=%s, hash=%s)",
        input_path,
        metadata.class_path,
        metadata.model_hash or "<missing>",
    )
    return model, metadata


def save_oof_predictions(
    oof_df: pd.DataFrame,
    output_path: str | Path,
) -> OOFArtifactMetadata:
    """
    Persist out-of-fold predictions to a parquet file with embedded metadata.

    Args:
        oof_df: OOF predictions DataFrame (expects ``prediction`` + ``fold_id`` columns).
        output_path: Destination ``.parquet`` file path.

    Returns:
        ``OOFArtifactMetadata`` describing the saved parquet artifact.

    Raises:
        ValueError: If the DataFrame schema is invalid.
        OSError: If parquet write fails (permissions, disk full, etc.).
    """
    output = Path(output_path)
    if output.suffix != ".parquet":
        raise ValueError(f"OOF artifact path must end with '.parquet' (got {output}).")

    prepared = _prepare_oof_frame(oof_df, copy_frame=True)
    metadata = _build_oof_metadata(prepared)

    table = _prepare_oof_table(prepared)
    existing_metadata = dict(table.schema.metadata or {})
    existing_metadata[OOF_METADATA_KEY] = metadata.to_json_bytes()
    table = table.replace_schema_metadata(existing_metadata)

    _ensure_parent_dir(output)
    _warn_on_overwrite(output)
    _atomic_parquet_write(table, output)

    logger.info(
        "Saved OOF predictions to %s (rows=%s, hash=%s)",
        output,
        metadata.num_rows,
        metadata.data_hash,
    )
    return metadata


def load_oof_predictions(
    input_path: str | Path,
    *,
    verify_hash: bool = True,
) -> tuple[pd.DataFrame, OOFArtifactMetadata]:
    """
    Load OOF predictions from parquet, optionally verifying integrity hash.

    Args:
        input_path: Path to parquet artifact.
        verify_hash: Whether to validate hash stored in metadata.

    Returns:
        Tuple of ``(DataFrame, metadata)``.

    Raises:
        FileNotFoundError: If path does not exist.
        OSError: If parquet read fails.
        ValueError: If hash validation fails or schema invalid.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"OOF predictions parquet not found at {input_path}.")

    try:
        table = pq.read_table(input_path)
    except Exception as exc:  # pragma: no cover - propagate with context
        raise OSError(f"Failed to read OOF parquet from {input_path}: {exc}") from exc

    raw_metadata = table.schema.metadata or {}
    metadata_bytes = raw_metadata.get(OOF_METADATA_KEY)
    if metadata_bytes is None:
        logger.warning("OOF parquet %s missing embedded metadata; recreating metadata.", input_path)
        dataframe = table.to_pandas()
        prepared = _prepare_oof_frame(dataframe, copy_frame=True)
        metadata = _build_oof_metadata(prepared)
        if verify_hash:
            # Hash verified implicitly by building from prepared frame
            pass
        return prepared, metadata

    metadata = OOFArtifactMetadata.from_json_bytes(metadata_bytes)

    dataframe = table.to_pandas()
    prepared = _prepare_oof_frame(dataframe, copy_frame=True)

    if verify_hash and metadata.data_hash:
        current_hash = _fingerprint_dataframe(prepared)
        if metadata.data_hash != current_hash:
            raise ValueError(
                f"OOF predictions hash mismatch for {input_path}. "
                f"Expected {metadata.data_hash}, observed {current_hash}."
            )

    logger.info(
        "Loaded OOF predictions from %s (rows=%s, hash=%s)",
        input_path,
        metadata.num_rows,
        metadata.data_hash or "<missing>",
    )
    return prepared, metadata
