"""
Real-time NEO Data Fetcher and Risk Predictor
Fixed version with correct column names and datetime handling
"""

import requests
import sqlite3
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timedelta, timezone
import logging
import os
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
NASA_API_KEY = os.getenv("NASA_API_KEY", "4znaUgLgFmi1vJanB3JG8h8I8zmL5mdQ2ZpIlQFO")
NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"
DB_PATH = "neo.db"
PRED_TABLE = "neo_predictions"
UPDATE_INTERVAL = 300
DAILY_REQUEST_LIMIT = 1000

# Model paths
XGBOOST_MODEL_PATH = "neo_hazard_model_final.joblib"
ISOLATION_FOREST_MODEL_PATH = "isolation_forest_model.pkl"


class NEODataFetcher:
    """Handles fetching data from NASA NeoWs API with rate limiting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.daily_request_count = 0
        self.daily_reset_time = datetime.now(timezone.utc)
        self.last_request_time = time.time()
    
    def _rate_limit_check(self):
        """Ensure we don't exceed rate limits"""
        current_time = datetime.now(timezone.utc)
        
        # Reset daily counter
        if current_time.date() > self.daily_reset_time.date():
            self.daily_request_count = 0
            self.daily_reset_time = current_time
        
        if self.daily_request_count >= DAILY_REQUEST_LIMIT:
            logger.warning("Daily API request limit reached")
            return False
        
        # Throttle: 6 seconds between requests
        elapsed = time.time() - self.last_request_time
        if elapsed < 6:
            time.sleep(6 - elapsed)
        
        return True
    
    def fetch_neos(self, start_date: str, end_date: str) -> Optional[Dict]:
        """Fetch NEO data for a date range"""
        if not self._rate_limit_check():
            return None
        
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "api_key": self.api_key
        }
        
        try:
            response = requests.get(NASA_NEO_URL, params=params, timeout=30)
            self.last_request_time = time.time()
            self.daily_request_count += 1
            
            if response.status_code == 200:
                logger.info(f"✓ Fetched data for {start_date} to {end_date}")
                return response.json()
            else:
                logger.error(f"API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return None


class NEOPredictor:
    """Handles ML predictions"""
    
    def __init__(self):
        self.xgboost_model = None
        self.isolation_forest = None
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            if os.path.exists(XGBOOST_MODEL_PATH):
                pipeline = joblib.load(XGBOOST_MODEL_PATH)
                self.xgboost_model = pipeline.get('model') or pipeline
                logger.info("✓ XGBoost model loaded")
            else:
                logger.warning(f"⚠ XGBoost model not found: {XGBOOST_MODEL_PATH}")
            
            if os.path.exists(ISOLATION_FOREST_MODEL_PATH):
                self.isolation_forest = joblib.load(ISOLATION_FOREST_MODEL_PATH)
                logger.info("✓ Isolation Forest loaded")
        
        except Exception as e:
            logger.error(f"Model loading error: {e}")
    
    def parse_neo_data(self, raw_data: Dict) -> pd.DataFrame:
        """Parse NASA API response"""
        records = []
        
        near_earth_objects = raw_data.get("near_earth_objects", {})
        
        for date_str, neos in near_earth_objects.items():
            for neo in neos:
                try:
                    # Diameter
                    diameter_data = neo.get("estimated_diameter", {}).get("meters", {})
                    diameter_min = diameter_data.get("estimated_diameter_min", 0)
                    diameter_max = diameter_data.get("estimated_diameter_max", 0)
                    diameter_m = (float(diameter_min) + float(diameter_max)) / 2 if diameter_min and diameter_max else 0
                    
                    # Close approach data
                    close_approach = neo.get("close_approach_data", [{}])[0]
                    miss_distance_km = float(close_approach.get("miss_distance", {}).get("kilometers", 0))
                    velocity_kmh = float(close_approach.get("relative_velocity", {}).get("kilometers_per_hour", 0))
                    velocity_kms = float(close_approach.get("relative_velocity", {}).get("kilometers_per_second", 0))
                    
                    record = {
                        "date": date_str,
                        "name": neo.get("name", "Unknown"),
                        "neo_reference_id": neo.get("neo_reference_id", ""),
                        "diameter_m": diameter_m,
                        "diameter_min_m": float(diameter_min) if diameter_min else 0,
                        "diameter_max_m": float(diameter_max) if diameter_max else 0,
                        "miss_distance_km": miss_distance_km,
                        "velocity_kmh": velocity_kmh,
                        "velocity_kms": velocity_kms,
                        "hazardous": int(neo.get("is_potentially_hazardous_asteroid", False)),
                        "absolute_magnitude_h": float(neo.get("absolute_magnitude_h", 0)),
                        "orbiting_body": close_approach.get("orbiting_body", "Earth")
                    }
                    
                    records.append(record)
                
                except Exception as e:
                    logger.warning(f"Parse error for {neo.get('name', 'Unknown')}: {e}")
                    continue
        
        return pd.DataFrame(records)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions"""
        if df.empty:
            return df
        
        # Simple risk calculation if models not available
        if self.xgboost_model is None and self.isolation_forest is None:
            # Fallback: rule-based risk
            df["risk_score"] = (
                (df["diameter_m"] > 140) * 0.4 +
                (df["velocity_kms"] > 15) * 0.3 +
                (df["miss_distance_km"] < 7480000) * 0.3
            )
            df["risk_label"] = pd.cut(
                df["risk_score"],
                bins=[0, 0.3, 0.7, 1.0],
                labels=["LOW", "MEDIUM", "HIGH"]
            )
        else:
            # Use actual models (simplified - adjust based on your model requirements)
            df["risk_score"] = df["hazardous"].astype(float)  # Placeholder
            df["risk_label"] = df["hazardous"].map({0: "LOW", 1: "HIGH"})
        
        # Fixed datetime - use timezone-aware
        df["prediction_time_utc"] = datetime.now(timezone.utc).isoformat()
        
        return df


class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database - matches your existing schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
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
            """)
            conn.commit()
        logger.info("✓ Database initialized")
    
    def get_last_fetch_date(self) -> Optional[str]:
        """Get most recent date in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT MAX(date) FROM {PRED_TABLE}")
            result = cursor.fetchone()
            return result[0] if result[0] else None
    
    def save_predictions(self, df: pd.DataFrame):
        """Save predictions, handling duplicates"""
        if df.empty:
            return
        
        save_cols = [
            "date", "name", "neo_reference_id", "diameter_m",
            "diameter_min_m", "diameter_max_m", "miss_distance_km",
            "velocity_kmh", "velocity_kms", "hazardous",
            "absolute_magnitude_h", "orbiting_body",
            "risk_score", "risk_label", "prediction_time_utc"
        ]
        
        df_save = df[[col for col in save_cols if col in df.columns]].copy()
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert, skipping duplicates
            for _, row in df_save.iterrows():
                try:
                    conn.execute(f"""
                        INSERT INTO {PRED_TABLE} 
                        ({', '.join(save_cols)})
                        VALUES ({', '.join(['?'] * len(save_cols))})
                    """, tuple(row[col] for col in save_cols))
                except sqlite3.IntegrityError:
                    # Duplicate - skip
                    pass
            
            conn.commit()
        
        logger.info(f"✓ Saved {len(df_save)} predictions")


def main():
    """Main loop for real-time updates"""
    logger.info("="*60)
    logger.info("NEO Real-time Updater Starting")
    logger.info("="*60)
    
    fetcher = NEODataFetcher(NASA_API_KEY)
    predictor = NEOPredictor()
    db_manager = DatabaseManager(DB_PATH)
    
    # Get last fetch date or start from today
    last_date = db_manager.get_last_fetch_date()
    
    if last_date:
        start_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        logger.info(f"Resuming from: {start_date.strftime('%Y-%m-%d')}")
    else:
        start_date = datetime.now()
        logger.info(f"Starting fresh: {start_date.strftime('%Y-%m-%d')}")
    
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            logger.info(f"\n--- Cycle {cycle_count} ---")
            
            # Fetch next 7 days
            current_start = start_date.strftime("%Y-%m-%d")
            current_end = (start_date + timedelta(days=6)).strftime("%Y-%m-%d")
            
            logger.info(f"Fetching: {current_start} to {current_end}")
            
            raw_data = fetcher.fetch_neos(current_start, current_end)
            
            if raw_data:
                df = predictor.parse_neo_data(raw_data)
                
                if not df.empty:
                    df_with_predictions = predictor.predict(df)
                    db_manager.save_predictions(df_with_predictions)
                    logger.info(f"Processed {len(df_with_predictions)} NEOs")
                else:
                    logger.info("No NEOs found for this period")
                
                start_date = start_date + timedelta(days=7)
                
                # If caught up to future, reset
                if start_date > datetime.now() + timedelta(days=7):
                    logger.info("Caught up! Checking for updates...")
                    start_date = datetime.now()
            else:
                logger.warning("Fetch failed, retrying next cycle")
            
            logger.info(f"Waiting {UPDATE_INTERVAL}s...")
            time.sleep(UPDATE_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("\n✓ Shutting down gracefully...")
            break
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    main()