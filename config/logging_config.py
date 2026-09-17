"""Logging setup using loguru."""

import sys
from loguru import logger
from config.settings import config


def setup_logging():
    """Configure loguru with console and file handlers."""
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=config.logging.log_level,
        colorize=True,
    )

    # File handler
    logger.add(
        config.logging.log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        enqueue=True,
    )

    return logger


# Singleton
logger = setup_logging()