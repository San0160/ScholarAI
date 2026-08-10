# AI generated code

import logging
from pathlib import Path

from ai_research_assistant.constants import LOG_DIR

# --------------------------------------------------------
# Load configuration
# --------------------------------------------------------
LOG_LEVEL = logging.INFO

# --------------------------------------------------------
# Create log directory
# --------------------------------------------------------

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

LOG_FILE = Path(LOG_DIR) / "application.log"

# --------------------------------------------------------
# Configure Logger
# --------------------------------------------------------

logger = logging.getLogger("AI_Research_Assistant")

# Prevent duplicate handlers (important for FastAPI/Uvicorn)
if not logger.handlers:

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False