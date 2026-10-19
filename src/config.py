"""Central settings. Everything can be overridden with environment variables."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
SCHEMA_DIR = ROOT / "schemas"
SQL_DIR = ROOT / "sql"

DB_URL = os.getenv("VIWELL_DB_URL", "mysql+pymysql://viwell:viwell@127.0.0.1:3306/viwell")

# --- synthetic data settings (Part 1) ---
SEED = 42
N_MEMBERS = 500
N_DAYS = 90
START_DATE = date(2026, 7, 1)          # day 1
# day 90 = 2026-09-28; the snapshot is "as of" this day.

# --- feature / ML windows (Parts 2-3) ---
FEATURE_DAYS = 76      # risk model sees days 1..76
LABEL_DAYS = 14        # ...and predicts days 77..90

# --- LLM service (Part 4) ---
LLM_BACKEND = os.getenv("VI_LLM_BACKEND", "ollama")        # "ollama" or "mock"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
