"""Logging setup: one logger that writes to the console and to <out_dir>/run.log."""
from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "sleep_detector"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(out_dir: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger. Safe to call more than once."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(module: str) -> logging.Logger:
    """Child logger for a module, e.g. get_logger(__name__)."""
    return logging.getLogger(f"{LOGGER_NAME}.{module.split('.')[-1]}")
