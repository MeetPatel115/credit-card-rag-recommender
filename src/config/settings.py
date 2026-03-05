from pathlib import Path
import os
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# data directories (will be created later)
DATA_DIR = PROJECT_ROOT / "data"
CARDS_DIR = DATA_DIR / "cards"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# model settings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_DB = os.getenv("VECTOR_DB", "chromadb")