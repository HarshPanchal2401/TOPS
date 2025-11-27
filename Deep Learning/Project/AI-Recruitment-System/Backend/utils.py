# backend/utils.py
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

def get_env(key, default=None):
    return os.environ.get(key, default)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = BASE_DIR / "backend_vector_store.faiss"
