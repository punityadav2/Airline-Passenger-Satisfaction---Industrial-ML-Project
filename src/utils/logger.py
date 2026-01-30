"""
Logging configuration for the Airline Passenger Satisfaction project.
Provides standardized logging across all modules.
"""

import logging
from pathlib import Path
from src.config import LOG_FORMAT, LOG_LEVEL, LOG_FILE


def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Setup and return a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional path to log file
        
    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    if log_file is None:
        log_file = LOG_FILE
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


# Create module-level logger
logger = setup_logger(__name__)
