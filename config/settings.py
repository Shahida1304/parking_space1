"""
Configuration settings for the Parking Space Detection project.
Loads environment variables from .env file and exposes them as constants.
Also supports st.secrets for Streamlit Cloud deployment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Project Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ── Helper: read from .env or st.secrets ───────────────────────────
def _get_secret(key: str, default: str = "") -> str:
    """Get config value from .env first, then st.secrets (Streamlit Cloud)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        sf = st.secrets.get("snowflake", {})
        mapping = {
            "SNOWFLAKE_ACCOUNT": "account",
            "SNOWFLAKE_USER": "user",
            "SNOWFLAKE_PASSWORD": "password",
            "SNOWFLAKE_DATABASE": "database",
            "SNOWFLAKE_SCHEMA": "schema",
            "SNOWFLAKE_WAREHOUSE": "warehouse",
            "SNOWFLAKE_ROLE": "role",
        }
        if key in mapping and mapping[key] in sf:
            return str(sf[mapping[key]])
    except Exception:
        pass
    return default


# ── Snowflake Configuration ────────────────────────────────────────
SNOWFLAKE_ACCOUNT = _get_secret("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER = _get_secret("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = _get_secret("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_DATABASE = _get_secret("SNOWFLAKE_DATABASE", "PARKING_DETECTION")
SNOWFLAKE_SCHEMA = _get_secret("SNOWFLAKE_SCHEMA", "PUBLIC")
SNOWFLAKE_WAREHOUSE = _get_secret("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_ROLE = _get_secret("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

# ── Model Configuration ───────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/best1.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))

# ── Output Configuration ──────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Class Mapping ─────────────────────────────────────────────────
# Class 0 = Empty/Free, Class 1 = Occupied
CLASS_NAMES = {0: "free", 1: "occupied"}
CLASS_COLORS = {
    "free": (0, 255, 0),        # Green in BGR
    "occupied": (0, 0, 255),    # Red in BGR
}

# ── Known Model Metrics (from training) ───────────────────────────
# These are the metrics obtained during training on the PKLot dataset.
# They will be seeded into Snowflake's MODEL_METRICS table.
TRAINING_METRICS = {
    "YOLO": {
        "precision": 0.997,
        "recall": 0.992,
        "map50": 0.995,
        "map50_95": 0.924,
        "speed_ms": 4.0,
    },
    "RT-DETR": {
        "precision": 0.996,
        "recall": 0.997,
        "map50": 0.995,
        "map50_95": 0.939,
        "speed_ms": 37.0,
    },
}
