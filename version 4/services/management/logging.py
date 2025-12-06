"""
EcoFlare Production Logging - Centralized for all services
Standards: RFC 5424 compliant, JSON optional, file rotation
"""
import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """
    Centralized logger factory for EcoFlare services

    Returns:
        Configured logger instance with file + console handlers
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    # Production config
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.setLevel(logging.INFO)

    # Console handler (teacher demo)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (production)
    file_handler = RotatingFileHandler(
        log_dir / "ecoflare.log", maxBytes=10*1024*1024, backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
