# AI generated code

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ai_research_assistant.constants import LOG_DIR

# --------------------------------------------------------
# Load configuration
# --------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
LOG_BACKUP_COUNT = 5             # keep 5 rotated files (25 MB total ceiling)

# --------------------------------------------------------
# Create log directory
# --------------------------------------------------------

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

LOG_FILE = Path(LOG_DIR) / "application.log"

# --------------------------------------------------------
# Configure Logger
# --------------------------------------------------------

logger = logging.getLogger("ai_research_assistant")

# Prevent duplicate handlers (important for FastAPI/Uvicorn)
if not logger.handlers:

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler (rotates at LOG_MAX_BYTES, keeps LOG_BACKUP_COUNT old files)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False