import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/ediss_data/ediss2/xai-texture/src/models/xai-model/logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(group_name, seed):
    log_path = LOG_DIR / f"{group_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(f"{group_name}_seed{seed}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear handlers if re-used across seeds

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Console handler (so you see it live in terminal / sbatch .out file)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger