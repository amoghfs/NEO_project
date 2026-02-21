"""
Real-time NEO Data Fetcher and Risk Predictor
Clean version with fixed model loading and date logic.
"""

import os
import time
import logging
from typing import Dict, Optional

import requests
import sqlite3
import pandas as pd
import numpy as np
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
UPDATE_INTERVAL = 60  # seconds between cycles

DAILY_REQUEST_LIMIT = 1000

XGBOOST_MODEL_PATH = "neo_hazard_model_xgb_iso.joblib"
ISOLATION_FOREST_MODEL_PATH = "neo_isolation_forest_model.joblib"

# If your pipeline uses this:
def log1p_array(X):
    # X is a numpy array from the imputer
    return np.log1p(X)

# ---------------------------------------------------------------------
# API Fetcher
# ---------------------------------------------------------------------
class NEODataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.daily_request_count = 0
        self.daily_reset_time = datetime.now(timezone.utc)
        self.last_request_time = time.time()

    def _rate_limit_check(self) -> bool:
        now = datetime.now(timezone.utc)

        # reset daily count
        if now.date() > self.daily_reset_time.date():
            self.daily_request_count = 0
            self.daily_reset_time = now

        if self.daily_request_count >= DAILY_REQUEST_LIMIT:
            logger.warning("Daily NASA API request limit reached.")
            return False

        # at least 6s between calls
        elapsed = time.time() - self.last_request_time
        if elapsed < 6:
            time.sleep(6 - elapsed)

        return True

    def fetch_neos(self, start_date: str, end_date: str) -> Optional[Dict]:
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
                logger.info(f"✓ Fetched data for {start_date} to {end_date}")
                return resp.json()
            else:
                logger.error(f"NASA API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as exc:
            logger.error(f"Fetch error: {exc}")
            return None


# ---------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------
class NEOPredictor:
    def __init__(self):
        self.xgboost_model = None
        self.isolation_forest = None
        self.load_models()

    def load_models(self) -> None:
        """Load the trained models (as pipelines) if they exist."""
        try:
            if os.path.exists(XGBOOST_MODEL_PATH):
                self.xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
                logger.info("✓ XGBoost pipeline loaded")
            else:
                logger.warning(f"⚠ XGBoost model not found: {XGBOOST_MODEL_PATH}")

            if os.path.exists(ISOLATION_FOREST_MODEL_PATH):
                self.isolation_forest = joblib.load(ISOLATION_FOREST_MODEL_PATH)
                logger.info("✓ Isolation Forest loaded")
            else:
                logger.warning(
                    f"⚠ Isolation Forest model not found: {ISOLATION_FOREST_MODEL_PATH}"
                )
        except Exception as exc:
            logger.error(f"Model loading error: {exc}")
            self.xgboost_model = None
            self.isolation_forest = None

    def parse_neo_data(self, raw_data: Dict) -> pd.DataFrame:
        records = []
        neo_by_date = raw_data.get("near_earth_objects", {})

        for date_str, neos in neo_by_date.items():
            for neo in neos:
                try:
                    diameter_data = neo.get("estimated_diameter", {}).get("meters", {})
                    d_min = float(diameter_data.get("estimated_diameter_min", 0) or 0)
                    d_max = float(diameter_data.get("estimated_diameter_max", 0) or 0)
                    diameter_m = (d_min + d_max) / 2 if d_min and d_max else 0.0

                    cad_list = neo.get("close_approach_data", [])
                    if not cad_list:
                        continue
                    close_approach = cad_list[0]

                    miss_distance_km = float(
                        close_approach.get("miss_distance", {}).get("kilometers", 0) or 0
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
                        f"Parse error for {neo.get('name', 'Unknown')}: {exc}"
                    )
                    continue

        return pd.DataFrame.from_records(records)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        # If you want to use the real model, plug the right feature list here.
        if self.xgboost_model is not None:
            # Example feature list – adjust to match how you trained
            feature_cols = [
                "absolute_magnitude_h",
                "diameter_m",
                "diameter_min_m",
                "diameter_max_m",
                "velocity_kmh",
                "velocity_kms",
                "miss_distance_km",
                "hazardous",
            ]
            X = df[feature_cols]
            try:
                proba = self.xgboost_model.predict_proba(X)[:, 1]
                df["risk_score"] = proba
            except Exception as exc:
                logger.error(f"Error using XGBoost model, falling back: {exc}")
                self.xgboost_model = None

        if self.xgboost_model is None:
            # Fallback rule-based risk
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

        df["prediction_time_utc"] = datetime.now(timezone.utc).isoformat()
        return df


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self) -> None:
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
                )
                """
            )
            conn.commit()
        logger.info("✓ Database initialized")

    def get_last_fetch_date(self) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(f"SELECT MAX(date) FROM {PRED_TABLE}")
            row = cur.fetchone()
            return row[0] if row and row[0] else None

    def save_predictions(self, df: pd.DataFrame) -> None:
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

        logger.info(f"✓ Saved {len(df_save)} predictions")


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("NEO Real-time Updater Starting")
    logger.info("=" * 60)

    fetcher = NEODataFetcher(NASA_API_KEY)
    predictor = NEOPredictor()
    db_manager = DatabaseManager(DB_PATH)

    last_date_str = db_manager.get_last_fetch_date()
    today = datetime.now()

    if last_date_str:
        from_db = datetime.strptime(last_date_str, "%Y-%m-%d") + timedelta(days=1)
        start_date = min(from_db, today)
        logger.info(f"Resuming from: {start_date.strftime('%Y-%m-%d')}")
    else:
        start_date = today
        logger.info(f"Starting fresh: {start_date.strftime('%Y-%m-%d')}")

    cycle_count = 0

    while True:
        try:
            cycle_count += 1
            logger.info("\n--- Cycle %d ---", cycle_count)

            current_start = start_date
            current_end = start_date + timedelta(days=6)

            start_str = current_start.strftime("%Y-%m-%d")
            end_str = current_end.strftime("%Y-%m-%d")

            logger.info(f"Fetching: {start_str} to {end_str}")

            raw_data = fetcher.fetch_neos(start_str, end_str)

            if raw_data:
                df = predictor.parse_neo_data(raw_data)
                if not df.empty:
                    df_pred = predictor.predict(df)
                    db_manager.save_predictions(df_pred)
                    logger.info(f"Processed {len(df_pred)} NEOs")
                else:
                    logger.info("No NEOs found for this period")

                # move forward one week
                start_date = start_date + timedelta(days=7)
            else:
                logger.warning("Fetch failed, retrying next cycle")

            logger.info(f"Waiting {UPDATE_INTERVAL}s...")
            time.sleep(UPDATE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n✓ Shutting down gracefully...")
            break
        except Exception as exc:
            logger.error(f"Error in main loop: {exc}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
