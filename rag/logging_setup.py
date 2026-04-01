"""Centralized logging configuration."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from rag.config import AppConfig


def setup_logging(config: AppConfig) -> logging.Logger:
    """Configure root logger with console + rotating file handlers."""
    log_cfg = config.logging
    log_dir = config.abs(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("rag")
    root.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))

    if root.handlers:
        return root  # already configured

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (skip Rich on Windows due to encoding issues)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "rag.log",
        maxBytes=log_cfg.max_bytes,
        backupCount=log_cfg.backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb.telemetry", "urllib3", "ollama"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root
