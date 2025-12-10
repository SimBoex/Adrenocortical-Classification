import os
import sys
import logging

def createLogger(name,onlyWarnings):
    logging_info = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

    log_dir = "logging"
    log_filepath = os.path.join(log_dir,name+"all_logs.log")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level= logging.INFO,
        format= logging_info,

        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if onlyWarnings: logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("Logger")

    return logger