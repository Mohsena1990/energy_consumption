"""
Logging utilities for CO2 forecasting framework.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory for log files
        run_id: Run identifier for log file naming
        level: Logging level
        console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger('co2_forecast')
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_file = log_dir / f"{run_id}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to: {log_file}")

    return logger


def get_logger(name: str = 'co2_forecast') -> logging.Logger:
    """Get the configured logger."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for logging sections."""

    def __init__(self, logger: logging.Logger, section: str):
        self.logger = logger
        self.section = section
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting: {self.section}")
        self.logger.info(f"{'='*60}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"Completed: {self.section} (elapsed: {elapsed})")
        else:
            self.logger.error(f"Failed: {self.section} - {exc_val}")
        self.logger.info(f"{'-'*60}")
        return False
