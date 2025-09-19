import os
import logging

def get_logger(log_dir, name: str = "segmentator"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Now configure with your settings
    log_file = os.path.join(log_dir, f"{name}.log")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    logger = logging.getLogger(f"{name}")
    return logger
