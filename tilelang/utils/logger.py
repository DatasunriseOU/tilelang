import logging
import sys

_logger_handlers_initialized = False

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger

def init_logger_handlers():
    """Initialize handlers for the root tilelang logger to avoid duplicate logs."""
    global _logger_handlers_initialized
    if _logger_handlers_initialized:
        return

    # Use the root tilelang logger to attach handlers
    logger = logging.getLogger("tilelang")
    
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
    
    file_handler = logging.FileHandler("tilelang.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    _logger_handlers_initialized = True
