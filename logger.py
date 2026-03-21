"""
Structured logging for the CHHAT analyzer.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "chhat", log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a structured logger with both console and file output."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / f"chhat_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the CHHAT logger (creates it if needed)."""
    logger = logging.getLogger("chhat")
    if not logger.handlers:
        return setup_logger()
    return logger
