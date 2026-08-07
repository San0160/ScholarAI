import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s"
)

project_name = "ai_research_assistant"

list_of_files = [

    # ---------------- GitHub ---------------- #
    ".github/workflows/.gitkeep",

    # ---------------- Source ---------------- #
    f"src/{project_name}/__init__.py",

    # API
    f"src/{project_name}/api/__init__.py",
    f"src/{project_name}/api/routes.py",

    # Configuration
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    # Core
    f"src/{project_name}/core/__init__.py",
    f"src/{project_name}/core/constants.py",

    # Document Ingestion
    f"src/{project_name}/ingestion/__init__.py",

    # Retrieval
    f"src/{project_name}/retrieval/__init__.py",

    # Embeddings
    f"src/{project_name}/embeddings/__init__.py",

    # Vector Database
    f"src/{project_name}/vector_store/__init__.py",

    # LLM
    f"src/{project_name}/llm/__init__.py",

    # Prompts
    f"src/{project_name}/prompts/__init__.py",

    # Memory
    f"src/{project_name}/memory/__init__.py",

    # Agents
    f"src/{project_name}/agents/__init__.py",

    # Evaluation
    f"src/{project_name}/evaluation/__init__.py",

    # Utilities
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    # Logging
    f"src/{project_name}/logging/__init__.py",

    # Pipeline
    f"src/{project_name}/workflows/__init__.py",

    # Entity
    f"src/{project_name}/entity/__init__.py",

    # Exceptions
    f"src/{project_name}/exception.py",

    # ---------------- Config Files ---------------- #
    "config/config.yaml",
    "params.yaml",

    # ---------------- Data ---------------- #
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/vector_db/.gitkeep",

    # ---------------- Docs ---------------- #
    "docs/.gitkeep",

    # ---------------- Deployment ---------------- #
    "deployment/.gitkeep",

    # ---------------- Tests ---------------- #
    "tests/__init__.py",

    # ---------------- Logs ---------------- #
    "logs/.gitkeep",

    # ---------------- Entry Points ---------------- #
    "app.py",
    "main.py",

    # ---------------- Project Files ---------------- #
    ".env",
    ".gitignore",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "README.md",

    # ---------------- Research ---------------- #
    "research/trials.ipynb",
]


for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w"):
            pass

        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")