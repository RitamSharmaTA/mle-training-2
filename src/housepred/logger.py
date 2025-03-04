import logging


def setup_logging(log_level="INFO", log_path=None, no_console_log=False):
    """
    Configures logging globally for the entire application.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_handlers = []

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        log_handlers.append(file_handler)

    if not no_console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        log_handlers.append(console_handler)

    logger = logging.getLogger("script")
    logger.setLevel(log_level.upper())

    logger.handlers.clear()

    for handler in log_handlers:
        logger.addHandler(handler)

    logger.propagate = False
