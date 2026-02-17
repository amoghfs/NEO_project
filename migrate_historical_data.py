"""
Historical Data Migration Script
Imports your existing 1975-2025 NEO data into the new database format
"""

import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import os

# Configuration
OLD_DB_PATH = "neo.db"  # Your existing database (change if different)
NEW_DB_PATH = "neo.db"  # New database with predictions
PRED_TABLE = "neo_predictions"

# Model paths
XGBOOST_MODEL_PATH = "xgboost_model.pkl"
ISOLATION_FOREST_MODEL_PATH = "isolation_forest_model.pkl"
SCALER_PATH = "scaler.pkl"


def load_historical_data(source_path):
    """
    Load historical data from various sources
    Supports: SQLite DB, CSV, or other formats
    """
    print(f"Loading historical data from {source_path}...")
    
    # Determine file type and load accordingly
    if source_path.endswith('.db'):
        # SQLite database
        with sqlite3.connect(source_path) as conn:
            # You may need to adjust table name and columns based on your schema
            # Common table names: 'neo_data', 'asteroids', 'near_earth_objects'
            
            # Try to detect the table
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Found tables: {tables}")
            
            # Use the first non-system table or prompt user
            if len(tables) == 1:
                table_name = tables[0]
            else:
                print("Multiple tables found. Which one contains NEO data?")
                for i, table in enumerate(tables):
                    print(f"{i+1}. {table}")
                choice = int(input("Enter number: ")) - 1
                table_name = tables[choice]
            
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    elif source_path.endswith('.csv'):
        # CSV file
        df = pd.read_csv(source_path)
    
    elif source_path.endswith('.parquet'):
        # Parquet file
        df = pd.read_parquet(source_path)
    
    else:
        print(f"Unsupported file type: {source_path}")
        return None
    
    print(f"Loaded {len(df):,} records")
    return df


def standardize_columns(df):
    """
    Standardize column names to match the new schema
    Adjust this function based on your existing column names
    """
    print("Standardizing column names...")
    
    # Common column name mappings
    # Adjust these based on your actual column names
    column_mapping = {
        # Date columns
        'close_approach_date': 'date',
        'approach_date': 'date',
        
        # Identification
        'designation': 'name',
        'neo_id': 'neo_reference_id',
        'reference_id': 'neo_reference_id',
        
        # Physical characteristics
        'estimated_diameter_m': 'diameter_m',
        'diameter': 'diameter_m',
        'est_diameter_meters': 'diameter_m',
        
        # Orbital parameters
        'miss_distance': 'miss_distance_km',
        'relative_velocity': 'velocity_kmh',
        'velocity': 'velocity_kmh',
        
        # Classification
        'is_hazardous': 'hazardous',
        'potentially_hazardous': 'hazardous',
        'is_potentially_hazardous_asteroid': 'hazardous',
        
        # Magnitude
        'absolute_magnitude_h': 'absolute_magnitude',
        'h_mag': 'absolute_magnitude',
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Handle diameter averaging if min/max are present
    if 'diameter_min' in df.columns and 'diameter_max' in df.columns:
        df['diameter_m'] = (df['diameter_min'] + df['diameter_max']) / 2
    
    # Ensure hazardous is 0/1
    if 'hazardous' in df.columns:
        df['hazardous'] = df['hazardous'].astype(int)
    
    # Add orbiting_body if missing (assume Earth)
    if 'orbiting_body' not in df.columns:
        df['orbiting_body'] = 'Earth'
    
    print(f"Columns after standardization: {df.columns.tolist()}")
    return df


def add_predictions(df):
    """
    Add predictions from both models to historical data
    """
    print("Loading models and generating predictions...")
    
    # Load models
    try:
        xgb_model = joblib.load(XGBOOST_MODEL_PATH) if os.path.exists(XGBOOST_MODEL_PATH) else None
        iso_model = joblib.load(ISOLATION_FOREST_MODEL_PATH) if os.path.exists(ISOLATION_FOREST_MODEL_PATH) else None
        scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    except Exception as e:
        print(f"Error loading models: {e}")
        return df
    
    # Feature engineering
    df['log_diameter'] = np.log1p(df['diameter_m'])
    df['log_miss_distance'] = np.log1p(df['miss_distance_km'])
    df['velocity_kms'] = df['velocity_kmh'] / 3600
    df['size_velocity_ratio'] = df['diameter_m'] / (df['velocity_kms'] + 1)
    df['proximity_index'] = df['diameter_m'] / (df['miss_distance_km'] + 1)
    
    # Define features
    feature_cols = [
        'diameter_m', 'miss_distance_km', 'velocity_kmh',
        'absolute_magnitude', 'log_diameter', 'log_miss_distance',
        'velocity_kms', 'size_velocity_ratio', 'proximity_index'
    ]
    
    # Fill missing features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[feature_cols].fillna(0)
    
    # Scale features
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values
    
    # XGBoost predictions
    if xgb_model is not None:
        print("Running XGBoost predictions...")
        df['xgb_risk_prob'] = xgb_model.predict_proba(X_scaled)[:, 1]
    else:
        print("XGBoost model not found, skipping...")
        df['xgb_risk_prob'] = 0.0
    
    # Isolation Forest predictions
    if iso_model is not None:
        print("Running Isolation Forest predictions...")
        anomaly_scores = iso_model.score_samples(X_scaled)
        
        # Normalize to 0-1 (higher = more anomalous)
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        if max_score - min_score > 0:
            df['isolation_anomaly_score'] = 1 - (anomaly_scores - min_score) / (max_score - min_score)
        else:
            df['isolation_anomaly_score'] = 0.0
        
        df['is_anomaly'] = (iso_model.predict(X_scaled) == -1).astype(int)
    else:
        print("Isolation Forest model not found, skipping...")
        df['isolation_anomaly_score'] = 0.0
        df['is_anomaly'] = 0
    
    # Combined risk score
    df['risk_score'] = (
        0.6 * df['xgb_risk_prob'] +
        0.4 * df['isolation_anomaly_score']
    )
    
    # Risk labels
    df['risk_label'] = pd.cut(
        df['risk_score'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    df['prediction_time_utc'] = datetime.utcnow().isoformat()
    
    print("Predictions completed!")
    return df


def save_to_database(df):
    """
    Save processed data to the new database
    """
    print("Saving to database...")
    
    # Select columns to save
    save_cols = [
        'date', 'name', 'neo_reference_id', 'orbiting_body',
        'diameter_m', 'miss_distance_km', 'velocity_kmh',
        'hazardous', 'absolute_magnitude', 'xgb_risk_prob',
        'isolation_anomaly_score', 'is_anomaly', 'risk_score',
        'risk_label', 'prediction_time_utc'
    ]
    
    # Filter to only columns that exist
    existing_cols = [col for col in save_cols if col in df.columns]
    df_save = df[existing_cols].copy()
    
    # Save to database
    with sqlite3.connect(NEW_DB_PATH) as conn:
        # Create table if it doesn't exist
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
        
        # Insert data, replacing duplicates
        df_save.to_sql(PRED_TABLE, conn, if_exists='append', index=False)
        
        # Remove duplicates
        conn.execute(f"""
            DELETE FROM {PRED_TABLE}
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM {PRED_TABLE}
                GROUP BY neo_reference_id
            )
        """)
        
        conn.commit()
        
        # Get final count
        cursor = conn.execute(f"SELECT COUNT(*) FROM {PRED_TABLE}")
        final_count = cursor.fetchone()[0]
    
    print(f"✅ Saved {final_count:,} records to database")


def main():
    """
    Main migration process
    """
    print("="*60)
    print("NEO Historical Data Migration")
    print("="*60 + "\n")
    
    # Get source file
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
    else:
        source_path = input("Enter path to historical data file (DB/CSV): ").strip()
    
    if not os.path.exists(source_path):
        print(f"❌ File not found: {source_path}")
        return 1
    
    # Load data
    df = load_historical_data(source_path)
    if df is None:
        return 1
    
    # Standardize columns
    df = standardize_columns(df)
    
    # Preview data
    print("\nData preview:")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
    
    confirm = input("\nDoes this look correct? (y/n): ").lower()
    if confirm != 'y':
        print("Migration cancelled. Adjust column mappings in standardize_columns() function.")
        return 1
    
    # Add predictions
    df = add_predictions(df)
    
    # Save to database
    save_to_database(df)
    
    print("\n" + "="*60)
    print("✅ Migration complete!")
    print("="*60)
    print("\nYou can now:")
    print("1. Run realtime_neo_updater.py to fetch new data")
    print("2. Run streamlit run enhanced_dashboard.py to view the dashboard")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
