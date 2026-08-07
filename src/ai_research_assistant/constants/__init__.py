from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]

CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"

CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"

RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
VECTOR_DB_PATH = DATA_DIR / "vector_db"
