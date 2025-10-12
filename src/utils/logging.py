"""
Centralized logging utilities for the quant-system.

Per Section 0 specs:
- JSON line format
- UTC timestamps
- INFO level by default, DEBUG on --debug flag
- Include: git_sha, config_hash, global_seed, region, split window IDs
"""

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines with required metadata."""

    def __init__(self, extra_fields: dict[str, Any] | None = None):
        """Initialize JSON formatter.

        Args:
            extra_fields: Additional fields to include in every log record
                (e.g., git_sha, config_hash, global_seed, region, split_window_ids)
        """
        super().__init__()
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON line.

        Args:
            record: The log record to format

        Returns:
            JSON string representation of the log record
        """
        # Use the actual event time, not the formatting time
        event_time = datetime.fromtimestamp(record.created, timezone.utc)

        log_data = {
            "timestamp": event_time.isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields (git_sha, config_hash, etc.)
        log_data.update(self.extra_fields)

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include any extra attributes added to the record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


def get_git_sha() -> str:
    """Get the current git SHA.

    Returns:
        Short git SHA (7 chars) or 'unknown' if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _normalize_for_hash(obj: Any) -> Any:
    """Normalize an object to a deterministic, JSON-serializable form.

    Args:
        obj: Object to normalize

    Returns:
        Normalized object safe for JSON serialization

    Raises:
        TypeError: If the object cannot be deterministically serialized
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(item) for item in obj]
    if isinstance(obj, set):
        # Sort to ensure determinism
        return sorted(_normalize_for_hash(item) for item in obj)

    # Fail fast for unknown types rather than using potentially non-deterministic str()
    msg = f"Cannot deterministically hash object of type {type(obj).__name__}"
    raise TypeError(msg)


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a deterministic hash of a configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Hex digest of the configuration (first 16 chars)

    Raises:
        TypeError: If config contains non-serializable types
    """
    normalized = _normalize_for_hash(config)
    config_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def setup_logging(
    level: str = "INFO",
    config: dict[str, Any] | None = None,
    global_seed: int = 42,
    region: str | None = None,
    split_window_ids: list[str] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> logging.Logger:
    """Set up JSON logging with required metadata fields.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Configuration dict to hash and include
        global_seed: Global random seed (default: 42 per specs)
        region: Trading region (US or CN)
        split_window_ids: Split window identifiers for walk-forward/CPCV
        extra_fields: Additional custom fields to include

    Returns:
        Configured root logger
    """
    # Build metadata fields
    metadata = {
        "git_sha": get_git_sha(),
        "global_seed": global_seed,
    }

    if config is not None:
        metadata["config_hash"] = compute_config_hash(config)

    if region is not None:
        metadata["region"] = region

    if split_window_ids is not None:
        metadata["split_window_ids"] = split_window_ids

    if extra_fields:
        metadata.update(extra_fields)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add JSON handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(extra_fields=metadata))
    root_logger.addHandler(handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
