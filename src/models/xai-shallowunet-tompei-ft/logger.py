import logging
from datetime import datetime
from pathlib import Path

import config

LOG_DIR = Path(config.LOG_DIR)
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(group_name, seed):
    log_path = LOG_DIR / f"{group_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(f"{group_name}_seed{seed}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger
