"""Configuración central del proyecto Predictor UFC.

Un solo lugar para rutas, URLs y parámetros, para que el resto del código no
tenga rutas ni constantes 'hardcodeadas'.
"""
from pathlib import Path

# --- Rutas del proyecto (relativas a este archivo, funcionan en cualquier PC) ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

# CSVs crudos que produce el scraper
EVENTS_CSV = RAW_DIR / "events.csv"
FIGHTS_CSV = RAW_DIR / "fights.csv"
FIGHT_STATS_CSV = RAW_DIR / "fight_stats.csv"
FIGHTERS_CSV = RAW_DIR / "fighters.csv"

# --- Scraping (ufcstats.com) ---
BASE_URL = "http://ufcstats.com"
EVENTS_LIST_URL = f"{BASE_URL}/statistics/events/completed?page=all"
REQUEST_DELAY_SEC = 0.6          # pausa cortés entre requests
REQUEST_TIMEOUT_SEC = 20
MAX_RETRIES = 3
USER_AGENT = "PredictorUFC/1.0 (proyecto educativo)"

# --- Reproducibilidad ---
RANDOM_SEED = 42
