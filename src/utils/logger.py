import logging
import colorlog


def setup_logger():
    # Define the log format
    log_format = "%(asctime)s - " "%(levelname)-8s - " "%(message)s"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    # Define the colors for each log level
    colorlog_format = f"{RESET}" f"{BOLD}" f"%(log_color)s" f"{log_format}"

    colorlog.basicConfig(format=colorlog_format)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the desired logging level here

    # Create console handler and set its logging level
    ch = colorlog.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Set the desired logging level here

    # Create formatter and add it to the handler
    formatter = colorlog.ColoredFormatter(
        colorlog_format,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    ch.setFormatter(formatter)

    # Add handlers to the logger
    # Check if logger already has handlers
    logger.addHandler(ch)

    return logger
