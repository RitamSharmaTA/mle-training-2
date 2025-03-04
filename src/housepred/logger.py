# src/logger.py
import logging


def setup_logging(log_level="INFO", log_path=None, no_console_log=False):
    """
    Configures logging globally for the application.

    Args:
        log_level (str): Logging level (e.g., DEBUG, INFO).
        log_path (str or None): Optional file path for logging.
        no_console_log (bool): If True, suppress console logging.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_handlers = []

    # File logging if log_path is provided
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        log_handlers.append(file_handler)

    # Console logging (unless suppressed)
    if not no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        log_handlers.append(console_handler)

    # Configure logging
    logging.basicConfig(level=log_level.upper(), handlers=log_handlers)
