"""
Real-time NEO Data Fetcher and Risk Predictor
Continuously polls NASA NeoWs API and updates predictions using XGBoost + Isolation Forest
"""

import requests
import sqlite3
import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime, timedelta
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
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")  # Get from environment or use DEMO_KEY
NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"
DB_PATH = "neo.db"
PRED_TABLE = "neo_predictions"
UPDATE_INTERVAL = 300  # Check for new data every 5 minutes (300 seconds)
DAILY_REQUEST_LIMIT = 1000
REQUESTS_PER_MINUTE = 10

# Model paths (adjust these to your actual model files)
XGBOOST_MODEL_PATH = "xgboost_model.pkl"
ISOLATION_FOREST_MODEL_PATH = "isolation_forest_model.pkl"
SCALER_PATH = "scaler.pkl"  # If you're using feature scaling


class NEODataFetcher:
    """Handles fetching data from NASA NeoWs API with rate limiting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.request_count = 0
        self.last_request_time = time.time()
        self.daily_request_count = 0
        self.daily_reset_time = datetime.now()
    
    def _rate_limit_check(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        
        # Reset daily counter if it's a new day
        if datetime.now().date() > self.daily_reset_time.date():
            self.daily_request_count = 0
            self.daily_reset_time = datetime.now()
        
        # Check daily limit
        if self.daily_request_count >= DAILY_REQUEST_LIMIT:
            logger.warning("Daily API request limit reached. Waiting until tomorrow.")
            return False
        
        # Check per-minute limit (simple throttling)
        if current_time - self.last_request_time < 6:  # 6 seconds between requests = 10/min
            time.sleep(6 - (current_time - self.last_request_time))
        
        return True
    
    def fetch_neos(self, start_date: str, end_date: str) -> Optional[Dict]:
        """
        Fetch NEO data for a date range
        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format
        Returns:
            Dict of NEO data or None if failed
        """
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
                logger.info(f"Successfully fetched NEO data for {start_date} to {end_date}")
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error fetching NEO data: {e}")
            return None


class NEOPredictor:
    """Handles ML predictions using XGBoost and Isolation Forest"""
    
    def __init__(self):
        self.xgboost_model = None
        self.isolation_forest = None
        self.scaler = None
        self.load_models()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists(XGBOOST_MODEL_PATH):
                self.xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
                logger.info("XGBoost model loaded successfully")
            else:
                logger.warning(f"XGBoost model not found at {XGBOOST_MODEL_PATH}")
            
            if os.path.exists(ISOLATION_FOREST_MODEL_PATH):
                self.isolation_forest = joblib.load(ISOLATION_FOREST_MODEL_PATH)
                logger.info("Isolation Forest model loaded successfully")
            else:
                logger.warning(f"Isolation Forest model not found at {ISOLATION_FOREST_MODEL_PATH}")
            
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def parse_neo_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Parse raw NASA API response into a DataFrame
        """
        records = []
        
        near_earth_objects = raw_data.get("near_earth_objects", {})
        
        for date_str, neos in near_earth_objects.items():
            for neo in neos:
                try:
                    # Extract diameter (average of min/max)
                    diameter_data = neo.get("estimated_diameter", {}).get("meters", {})
                    diameter_min = diameter_data.get("estimated_diameter_min", 0)
                    diameter_max = diameter_data.get("estimated_diameter_max", 0)
                    diameter_m = (diameter_min + diameter_max) / 2 if diameter_min and diameter_max else 0
                    
                    # Get close approach data (use first approach)
                    close_approach = neo.get("close_approach_data", [{}])[0]
                    
                    miss_distance_km = float(close_approach.get("miss_distance", {}).get("kilometers", 0))
                    velocity_kmh = float(close_approach.get("relative_velocity", {}).get("kilometers_per_hour", 0))
                    orbiting_body = close_approach.get("orbiting_body", "Earth")
                    
                    record = {
                        "date": date_str,
                        "name": neo.get("name", "Unknown"),
                        "neo_reference_id": neo.get("neo_reference_id", ""),
                        "orbiting_body": orbiting_body,
                        "diameter_m": diameter_m,
                        "miss_distance_km": miss_distance_km,
                        "velocity_kmh": velocity_kmh,
                        "hazardous": int(neo.get("is_potentially_hazardous_asteroid", False)),
                        "absolute_magnitude": float(neo.get("absolute_magnitude_h", 0))
                    }
                    
                    records.append(record)
                
                except Exception as e:
                    logger.warning(f"Error parsing NEO {neo.get('name', 'Unknown')}: {e}")
                    continue
        
        return pd.DataFrame(records)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML models
        Adjust this based on what features your models were trained on
        """
        df = df.copy()
        
        # Basic feature engineering
        df["log_diameter"] = np.log1p(df["diameter_m"])
        df["log_miss_distance"] = np.log1p(df["miss_distance_km"])
        df["velocity_kms"] = df["velocity_kmh"] / 3600  # Convert to km/s
        
        # Add more features based on your training
        # Example: interaction features
        df["size_velocity_ratio"] = df["diameter_m"] / (df["velocity_kms"] + 1)
        df["proximity_index"] = df["diameter_m"] / (df["miss_distance_km"] + 1)
        
        return df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions using both models
        """
        if df.empty:
            return df
        
        df = self.create_features(df)
        
        # Define feature columns (adjust based on your training)
        feature_cols = [
            "diameter_m", "miss_distance_km", "velocity_kmh", 
            "absolute_magnitude", "log_diameter", "log_miss_distance",
            "velocity_kms", "size_velocity_ratio", "proximity_index"
        ]
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        X = df[feature_cols].fillna(0)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # XGBoost prediction
        if self.xgboost_model is not None:
            try:
                xgb_probs = self.xgboost_model.predict_proba(X_scaled)[:, 1]
                df["xgb_risk_prob"] = xgb_probs
            except Exception as e:
                logger.error(f"XGBoost prediction error: {e}")
                df["xgb_risk_prob"] = 0.0
        else:
            df["xgb_risk_prob"] = 0.0
        
        # Isolation Forest anomaly detection
        if self.isolation_forest is not None:
            try:
                # Returns -1 for outliers, 1 for inliers
                anomaly_labels = self.isolation_forest.predict(X_scaled)
                # Get anomaly scores (lower = more anomalous)
                anomaly_scores = self.isolation_forest.score_samples(X_scaled)
                
                # Convert to 0-1 scale (higher = more anomalous)
                # Normalize scores to [0, 1]
                min_score = anomaly_scores.min()
                max_score = anomaly_scores.max()
                if max_score - min_score > 0:
                    normalized_scores = 1 - (anomaly_scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.zeros_like(anomaly_scores)
                
                df["isolation_anomaly_score"] = normalized_scores
                df["is_anomaly"] = (anomaly_labels == -1).astype(int)
            except Exception as e:
                logger.error(f"Isolation Forest prediction error: {e}")
                df["isolation_anomaly_score"] = 0.0
                df["is_anomaly"] = 0
        else:
            df["isolation_anomaly_score"] = 0.0
            df["is_anomaly"] = 0
        
        # Combined risk score (weighted average)
        # Adjust weights based on your preference
        df["risk_score"] = (
            0.6 * df["xgb_risk_prob"] + 
            0.4 * df["isolation_anomaly_score"]
        )
        
        # Risk labels based on thresholds
        df["risk_label"] = pd.cut(
            df["risk_score"],
            bins=[0, 0.3, 0.7, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"]
        )
        
        df["prediction_time_utc"] = datetime.utcnow().isoformat()
        
        return df


class DatabaseManager:
    """Handles database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    name TEXT,
                    neo_reference_id TEXT UNIQUE,
                    orbiting_body TEXT,
                    diameter_m REAL,
                    miss_distance_km REAL,
                    velocity_kmh REAL,
                    hazardous INTEGER,
                    absolute_magnitude REAL,
                    xgb_risk_prob REAL,
                    isolation_anomaly_score REAL,
                    is_anomaly INTEGER,
                    risk_score REAL,
                    risk_label TEXT,
                    prediction_time_utc TEXT
                )
            """)
            conn.commit()
        logger.info("Database initialized")
    
    def get_last_fetch_date(self) -> Optional[str]:
        """Get the most recent date we have data for"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT MAX(date) FROM {PRED_TABLE}")
            result = cursor.fetchone()
            return result[0] if result[0] else None
    
    def save_predictions(self, df: pd.DataFrame):
        """Save predictions to database, handling duplicates"""
        if df.empty:
            return
        
        # Select columns to save
        save_cols = [
            "date", "name", "neo_reference_id", "orbiting_body",
            "diameter_m", "miss_distance_km", "velocity_kmh",
            "hazardous", "absolute_magnitude", "xgb_risk_prob",
            "isolation_anomaly_score", "is_anomaly", "risk_score",
            "risk_label", "prediction_time_utc"
        ]
        
        df_save = df[[col for col in save_cols if col in df.columns]].copy()
        
        with sqlite3.connect(self.db_path) as conn:
            # Use REPLACE to handle duplicates based on neo_reference_id
            df_save.to_sql(PRED_TABLE, conn, if_exists="append", index=False)
            
            # Remove duplicates, keeping the latest prediction
            conn.execute(f"""
                DELETE FROM {PRED_TABLE}
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM {PRED_TABLE}
                    GROUP BY neo_reference_id
                )
            """)
            conn.commit()
        
        logger.info(f"Saved {len(df_save)} predictions to database")


def main():
    """Main loop for real-time updates"""
    logger.info("Starting NEO Real-time Updater")
    
    # Initialize components
    fetcher = NEODataFetcher(NASA_API_KEY)
    predictor = NEOPredictor()
    db_manager = DatabaseManager(DB_PATH)
    
    # Get last fetch date or start from today
    last_date = db_manager.get_last_fetch_date()
    
    if last_date:
        # Start from the day after last fetch
        start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1))
    else:
        # Start from today if no data exists
        start_date = datetime.now()
    
    logger.info(f"Starting from date: {start_date.strftime('%Y-%m-%d')}")
    
    while True:
        try:
            # Fetch data for the next 7 days (NASA API limit)
            current_start = start_date.strftime("%Y-%m-%d")
            current_end = (start_date + timedelta(days=6)).strftime("%Y-%m-%d")
            
            logger.info(f"Fetching NEO data from {current_start} to {current_end}")
            
            raw_data = fetcher.fetch_neos(current_start, current_end)
            
            if raw_data:
                # Parse and predict
                df = predictor.parse_neo_data(raw_data)
                
                if not df.empty:
                    df_with_predictions = predictor.predict(df)
                    db_manager.save_predictions(df_with_predictions)
                    logger.info(f"Processed {len(df_with_predictions)} NEOs")
                else:
                    logger.info("No new NEO data found for this period")
                
                # Move to next week
                start_date = start_date + timedelta(days=7)
                
                # If we've reached the future, reset to check for updates
                if start_date > datetime.now() + timedelta(days=7):
                    logger.info("Caught up to current data. Checking for updates...")
                    start_date = datetime.now()
            
            else:
                logger.warning("Failed to fetch data, retrying in next cycle")
            
            # Wait before next update
            logger.info(f"Waiting {UPDATE_INTERVAL} seconds before next update...")
            time.sleep(UPDATE_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Shutting down updater...")
            break
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying


if __name__ == "__main__":
    main()
