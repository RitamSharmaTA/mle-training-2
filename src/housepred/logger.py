import logging
import sys


def setup_logging(log_level="INFO", log_path=None, no_console_log=False):

    log_format = "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_handlers = []

    print(f"Setting up logging with level: {log_level}")

    if log_path:
        file_handler = logging.FileHandler(log_path, mode="w")  # Overwrites log file
        file_handler.setFormatter(logging.Formatter(log_format))
        log_handlers.append(file_handler)

    # Console logging (GitHub Actions terminal)
    console_handler = logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout
    console_handler.setFormatter(logging.Formatter(log_format))
    log_handlers.append(console_handler)

    logging.basicConfig(
        level=log_level.upper(), handlers=log_handlers, format=log_format
    )

    logging.getLogger().info("Logging configured successfully")
