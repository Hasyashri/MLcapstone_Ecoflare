"""import logging
import os

def setup_logger(name="ecoflare"):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(f"{log_dir}/ecoflare.log")
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger"""


import logging
import os

def setup_logger(name="ecoflare"):

    # Ensure logs/ directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Standard log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File log output
    file_handler = logging.FileHandler(f"{log_dir}/ecoflare.log")
    file_handler.setFormatter(formatter)

    # Console output (shows in Docker Desktop logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Prevent duplicate logs
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
