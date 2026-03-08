"""
FightMind AI — Centralized Logging Configuration
==================================================
Single source of truth for all logging across the service.

Strategy:
  - LOCAL  (ENV=development): human-readable colored console output
  - PROD   (ENV=production) : structured JSON logs → consumed by Render.com

Per-module log levels are controlled via logging.yaml (project root).
Edit that file to change verbosity without touching any Python code.

Usage anywhere in the codebase:
    from src.core.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("My message", extra={"query_id": "abc123"})
"""

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # pyyaml

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_NAME = "fightmind-model"
ENV = os.getenv("ENV", "development")
IS_PROD = ENV == "production"

# Path to the YAML config — always relative to the project root (fightmind-model/)
# Path breakdown from src/core/logging_config.py:
#   parents[0] = src/core
#   parents[1] = src
#   parents[2] = fightmind-model  ← project root ✓
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGING_CONFIG_PATH = _PROJECT_ROOT / "logging.yaml"

# Fallback global level if YAML config is missing or unreadable
_FALLBACK_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
_FALLBACK_LEVEL = getattr(logging, _FALLBACK_LEVEL_STR, logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for local development.
    Output: 2026-03-08 12:00:00 | INFO     | data_collection.scraper | Message [key=val]
    """

    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Green
        "WARNING":  "\033[33m",   # Yellow
        "ERROR":    "\033[31m",   # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    _SKIP_FIELDS = frozenset({
        "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno",
        "funcName", "created", "msecs", "relativeCreated", "thread",
        "threadName", "processName", "process", "name", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        # Shorten the module name for readability (strip "src." prefix)
        name = record.name.replace("src.", "")
        color = self.LEVEL_COLORS.get(record.levelname, "")
        level_str = f"{color}{record.levelname:<8}{self.RESET}"
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        line = f"{ts} | {level_str} | {name:<30} | {record.getMessage()}"

        # Append exception traceback if present
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        # Append any extra context fields passed via extra={}
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in self._SKIP_FIELDS and not k.startswith("_")
        }
        if extra:
            ctx = "  ".join(f"{k}={v}" for k, v in extra.items())
            line += f"  [{ctx}]"

        return line


class JsonFormatter(logging.Formatter):
    """
    Structured JSON formatter for production deployment on Render.com.

    Every log line is a single JSON object. Render.com aggregates these
    into its dashboard where you can filter by level, service, or any field.

    Example output:
    {
      "timestamp": "2026-03-08T06:30:00.123Z",
      "level": "INFO",
      "service": "fightmind-model",
      "logger": "src.data_collection.scraper",
      "message": "Wikipedia scraper complete",
      "collected": 174,
      "skipped": 6
    }
    """

    _SKIP_FIELDS = frozenset({
        "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno",
        "funcName", "created", "msecs", "relativeCreated", "thread",
        "threadName", "processName", "process", "name", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "service": SERVICE_NAME,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include exception as structured fields (not a raw traceback string)
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Include extra context fields
        for k, v in record.__dict__.items():
            if k not in self._SKIP_FIELDS and not k.startswith("_"):
                payload[k] = v

        return json.dumps(payload, default=str, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# YAML Config Loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml_config() -> dict:
    """
    Load logging.yaml from the project root.

    Returns the parsed YAML dict, or an empty dict if the file is missing
    or malformed (with a printed warning — logging isn't ready yet here).
    """
    if not LOGGING_CONFIG_PATH.exists():
        print(
            f"[logging_config] WARNING: {LOGGING_CONFIG_PATH} not found. "
            f"Using fallback level: {_FALLBACK_LEVEL_STR}",
            file=sys.stderr,
        )
        return {}

    try:
        with open(LOGGING_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(
            f"[logging_config] ERROR: Failed to parse {LOGGING_CONFIG_PATH}: {exc}. "
            f"Using fallback level: {_FALLBACK_LEVEL_STR}",
            file=sys.stderr,
        )
        return {}


def _level_key() -> str:
    """Return the YAML level key for the current environment."""
    return "production_level" if IS_PROD else "development_level"


# ─────────────────────────────────────────────────────────────────────────────
# Setup — Called Once at Application Startup
# ─────────────────────────────────────────────────────────────────────────────

_configured = False   # guard: only configure once per process


def setup_logging() -> None:
    """
    Configure all loggers from logging.yaml.
    Call this ONCE at startup (in main.py or a script __main__ block).

    What it does:
      1. Reads logging.yaml from the project root
      2. Sets the root logger to the global default level
      3. Overrides each module listed under `loggers:` with its configured level
      4. Attaches the appropriate formatter (human-readable or JSON)

    After this call, `get_logger(__name__)` in any module will automatically
    pick up the level set in logging.yaml for that module path.
    """
    global _configured
    if _configured:
        return
    _configured = True

    config = _load_yaml_config()
    level_key = _level_key()

    # ── Step 1: Configure the root logger (global default) ────────────────────
    root_cfg = config.get("root", {})
    root_level_str = root_cfg.get(level_key, _FALLBACK_LEVEL_STR).upper()
    root_level = getattr(logging, root_level_str, _FALLBACK_LEVEL)

    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)
    root_logger.handlers.clear()

    # ── Step 2: Attach formatter ───────────────────────────────────────────────
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(root_level)
    handler.setFormatter(JsonFormatter() if IS_PROD else HumanReadableFormatter())
    root_logger.addHandler(handler)

    # ── Step 3: Apply per-module overrides from logging.yaml ──────────────────
    module_configs: dict = config.get("loggers", {})
    for module_name, module_cfg in module_configs.items():
        level_str = module_cfg.get(level_key, root_level_str).upper()
        level = getattr(logging, level_str, root_level)
        logging.getLogger(module_name).setLevel(level)

    # ── Step 4: Log startup confirmation ──────────────────────────────────────
    startup_logger = logging.getLogger("fightmind.startup")
    startup_logger.info(
        "Logging initialised from config file",
        extra={
            "config_file": str(LOGGING_CONFIG_PATH),
            "env": ENV,
            "root_level": root_level_str,
            "module_overrides": len(module_configs),
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Must call setup_logging() first at startup.

    The logger name should always be __name__ of the calling module so that
    it matches the dotted path in logging.yaml.

    Args:
        name: module name, e.g. __name__ → "src.data_collection.scraper"

    Returns:
        logging.Logger configured with the level from logging.yaml

    Example:
        from src.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Scraping started", extra={"topic_count": 180})
    """
    return logging.getLogger(name)
