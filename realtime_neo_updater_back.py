"""
Real-time NEO Data Fetcher and Risk Predictor
Rewritten version with correct schema, datetime handling, and robust DB inserts.
"""

import os
import time
import logging
from typing import Dict, Optional

import requests
import sqlite3
import pandas as pd
import joblib
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
NASA_API_KEY = os.getenv("NASA_API_KEY", "4znaUgLgFmi1vJanB3JG8h8I8zmL5mdQ2ZpIlQFO")
NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"
DB_PATH = "neo.db"

PRED_TABLE = "neo_predictions"
UPDATE_INTERVAL = 300           # seconds between cycles
DAILY_REQUEST_LIMIT = 1000      # safety cap for NASA API calls

XGBOOST_MODEL_PATH = "neo_hazard_model_xgb_iso.joblib"
ISOLATION_FOREST_MODEL_PATH = "neo_isolation_forest_model.joblib"

# ---------------------------------------------------------------------
# API Fetcher
# ---------------------------------------------------------------------
class NEODataFetcher:
    """Fetches NEO data from NASA NeoWs with simple rate limiting."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.daily_request_count = 0
        self.daily_reset_time = datetime.now(timezone.utc)
        self.last_request_time = time.time()

    def _rate_limit_check(self) -> bool:
        """Enforce daily and per‑request limits."""
        now = datetime.now(timezone.utc)

        # Reset daily count at UTC day boundary
        if now.date() > self.daily_reset_time.date():
            self.daily_request_count = 0
            self.daily_reset_time = now

        if self.daily_request_count >= DAILY_REQUEST_LIMIT:
            logger.warning("Daily NASA API request limit reached.")
            return False

        # Enforce at least 6 seconds between calls (NASA default is generous,
        # this is just a guard).
        elapsed = time.time() - self.last_request_time
        if elapsed < 6:
            time.sleep(6 - elapsed)

        return True

    def fetch_neos(self, start_date: str, end_date: str) -> Optional[Dict]:
        """Fetch NEO feed JSON for the given date range (YYYY-MM-DD)."""
        if not self._rate_limit_check():
            return None

        params = {
            "start_date": start_date,
            "end_date": end_date,
            "api_key": self.api_key,
        }

        try:
            resp = requests.get(NASA_NEO_URL, params=params, timeout=30)
            self.last_request_time = time.time()
            self.daily_request_count += 1

            if resp.status_code == 200:
                logger.info(f"Fetched NEO data for {start_date} to {end_date}.")
                return resp.json()
            else:
                logger.error(f"NASA API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as exc:
            logger.error(f"Error fetching NEO data: {exc}")
            return None

# ---------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------
class NEOPredictor:
    """Loads ML models (if present) and computes risk scores."""

    def __init__(self):
        self.xgboost_model = None
        self.isolation_forest = None
        self._load_models()

    def _load_models(self) -> None:
        """Load saved models from disk, if available."""
        try:
            if os.path.exists(XGBOOST_MODEL_PATH):
                pipeline = joblib.load(XGBOOST_MODEL_PATH)
                # If you saved a sklearn Pipeline, adjust this line as needed.
                self.xgboost_model = getattr(pipeline, "predict_proba", pipeline)
                logger.info("XGBoost model loaded.")
            else:
                logger.warning(f"XGBoost model not found at {XGBOOST_MODEL_PATH}.")

            if os.path.exists(ISOLATION_FOREST_MODEL_PATH):
                self.isolation_forest = joblib.load(ISOLATION_FOREST_MODEL_PATH)
                logger.info("Isolation Forest model loaded.")
            else:
                logger.warning(
                    f"Isolation Forest model not found at {ISOLATION_FOREST_MODEL_PATH}."
                )
        except Exception as exc:
            logger.error(f"Error loading models: {exc}")

    def parse_neo_data(self, raw_data: Dict) -> pd.DataFrame:
        """Convert NASA NEO feed JSON into a flat DataFrame."""
        records = []

        neo_by_date = raw_data.get("near_earth_objects", {})
        for date_str, neos in neo_by_date.items():
            for neo in neos:
                try:
                    # Diameter (meters)
                    diameter_data = (
                        neo.get("estimated_diameter", {}).get("meters", {})
                    )
                    d_min = float(diameter_data.get("estimated_diameter_min", 0) or 0)
                    d_max = float(diameter_data.get("estimated_diameter_max", 0) or 0)
                    diameter_m = (d_min + d_max) / 2 if d_min and d_max else 0.0

                    # Close approach data
                    cad_list = neo.get("close_approach_data", [])
                    if not cad_list:
                        continue
                    close_approach = cad_list[0]

                    miss_distance_km = float(
                        close_approach.get("miss_distance", {})
                        .get("kilometers", 0)
                        or 0
                    )
                    velocity_kmh = float(
                        close_approach.get("relative_velocity", {})
                        .get("kilometers_per_hour", 0)
                        or 0
                    )
                    velocity_kms = float(
                        close_approach.get("relative_velocity", {})
                        .get("kilometers_per_second", 0)
                        or 0
                    )

                    record = {
                        "date": date_str,
                        "name": neo.get("name", "Unknown"),
                        "neo_reference_id": neo.get("neo_reference_id", ""),
                        "diameter_m": diameter_m,
                        "diameter_min_m": d_min,
                        "diameter_max_m": d_max,
                        "miss_distance_km": miss_distance_km,
                        "velocity_kmh": velocity_kmh,
                        "velocity_kms": velocity_kms,
                        "hazardous": int(
                            bool(neo.get("is_potentially_hazardous_asteroid", False))
                        ),
                        "absolute_magnitude_h": float(
                            neo.get("absolute_magnitude_h", 0) or 0
                        ),
                        "orbiting_body": close_approach.get("orbiting_body", "Earth"),
                    }

                    records.append(record)
                except Exception as exc:
                    logger.warning(
                        f"Error parsing NEO {neo.get('name', 'Unknown')}: {exc}"
                    )
                    continue

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk_score and risk_label columns to the DataFrame."""
        if df.empty:
            return df

        # Simple rules if models are absent
        if self.xgboost_model is None and self.isolation_forest is None:
            df["risk_score"] = (
                (df["diameter_m"] > 140).astype(float) * 0.4
                + (df["velocity_kms"] > 15).astype(float) * 0.3
                + (df["miss_distance_km"] < 7_480_000).astype(float) * 0.3
            )

            df["risk_label"] = pd.cut(
                df["risk_score"],
                bins=[0.0, 0.3, 0.7, 1.0],
                labels=["LOW", "MEDIUM", "HIGH"],
                include_lowest=True,
            )
        else:
            # Placeholder: use your trained model’s actual feature set here.
            df["risk_score"] = df["hazardous"].astype(float)
            df["risk_label"] = df["hazardous"].map({0: "LOW", 1: "HIGH"})

        # Timestamp for this prediction batch (UTC, ISO 8601)
        df["prediction_time_utc"] = datetime.now(timezone.utc).isoformat()

        return df

# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------
class DatabaseManager:
    """Creates and maintains the neo_predictions table and writes rows."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create prediction table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    name TEXT,
                    neo_reference_id TEXT,
                    diameter_m REAL,
                    diameter_min_m REAL,
                    diameter_max_m REAL,
                    miss_distance_km REAL,
                    velocity_kmh REAL,
                    velocity_kms REAL,
                    hazardous INTEGER,
                    absolute_magnitude_h REAL,
                    orbiting_body TEXT,
                    risk_score REAL,
                    risk_label TEXT,
                    prediction_time_utc TEXT,
                    UNIQUE(neo_reference_id, date)
                );
                """
            )
            conn.commit()
        logger.info("Database initialized / verified.")

    def get_last_fetch_date(self) -> Optional[str]:
        """Return the max date (YYYY-MM-DD) stored in the predictions table."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"SELECT MAX(date) FROM {PRED_TABLE}")
            row = cur.fetchone()
            return row[0] if row and row[0] else None

    def save_predictions(self, df: pd.DataFrame) -> None:
        """Insert prediction rows, skipping duplicates via UNIQUE constraint."""
        if df.empty:
            return

        save_cols = [
            "date",
            "name",
            "neo_reference_id",
            "diameter_m",
            "diameter_min_m",
            "diameter_max_m",
            "miss_distance_km",
            "velocity_kmh",
            "velocity_kms",
            "hazardous",
            "absolute_magnitude_h",
            "orbiting_body",
            "risk_score",
            "risk_label",
            "prediction_time_utc",
        ]

        df_save = df[save_cols].copy()

        placeholders = ", ".join(["?"] * len(save_cols))
        col_list = ", ".join(save_cols)

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                f"""
                INSERT OR IGNORE INTO {PRED_TABLE} ({col_list})
                VALUES ({placeholders})
                """,
                [
                    tuple(row[col] for col in save_cols)
                    for _, row in df_save.iterrows()
                ],
            )
            conn.commit()

        logger.info(f"Saved {len(df_save)} prediction rows to database.")

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting NEO real-time updater")
    logger.info("=" * 60)

    fetcher = NEODataFetcher(NASA_API_KEY)
    predictor = NEOPredictor()
    db = DatabaseManager(DB_PATH)

    # Determine where to start
    last_date = db.get_last_fetch_date()
    if last_date:
        start_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        logger.info(f"Resuming from {start_date.strftime('%Y-%m-%d')}.")
    else:
        start_date = datetime.now()
        logger.info(f"Starting fresh from {start_date.strftime('%Y-%m-%d')}.")

    cycle = 0

    while True:
        try:
            cycle += 1
            logger.info("")
            logger.info(f"--- Cycle {cycle} ---")

            # NeoWs feed supports up to 7 days at once
            period_start = start_date
            period_end = start_date + timedelta(days=6)

            start_str = period_start.strftime("%Y-%m-%d")
            end_str = period_end.strftime("%Y-%m-%d")
            logger.info(f"Requesting NEOs from {start_str} to {end_str}.")

            raw = fetcher.fetch_neos(start_str, end_str)

            if raw is not None:
                df = predictor.parse_neo_data(raw)
                if not df.empty:
                    df_pred = predictor.predict(df)
                    db.save_predictions(df_pred)
                    logger.info(f"Processed {len(df_pred)} NEOs.")
                else:
                    logger.info("No NEOs returned for this period.")

                start_date = start_date + timedelta(days=7)

                # If we move too far into the future, reset to "today"
                if start_date > datetime.now() + timedelta(days=7):
                    logger.info("Caught up to future window; resetting to today.")
                    start_date = datetime.now()
            else:
                logger.warning("Fetch failed; will retry next cycle.")

            logger.info(f"Sleeping for {UPDATE_INTERVAL} seconds.")
            time.sleep(UPDATE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Shutdown requested. Exiting main loop.")
            break
        except Exception as exc:
            logger.error(f"Unexpected error in main loop: {exc}", exc_info=True)
            time.sleep(60)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
