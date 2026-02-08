from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """
    Create a run-scoped logger that logs to both console and file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    logger = logging.getLogger("edainsights")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

