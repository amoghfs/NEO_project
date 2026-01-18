import pandas as pd
import joblib
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
#
# ---------- FIX: Added the missing function ----------
#
# Your saved model ('neo_hazard_model.joblib') was trained using
# this custom function. To load the model, this function
# must be defined in the script first.
#
def log1p_array(X):
    """Applies the numpy log1p transformation."""
    # Assuming this is the function from your training script:
    return np.log1p(X)
#
# -----------------------------------------------------
#


# ---------- 1) Config ----------
DB_PATH = "neo.db"                  # same DB file
MODEL_PATH = "neo_hazard_model.joblib"
MODEL_VERSION = "logreg_v1"         # bump when you retrain

# ---------- 2) Connect to SQLite & load model ----------
engine = create_engine(f"sqlite:///{DB_PATH}")

# ---------- FIX: Create the scores table if it doesn't exist ----------
# This ensures the table exists before we try to read from it.

create_table_sql = text("""
CREATE TABLE IF NOT EXISTS neo_risk_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    neo_row_id INTEGER NOT NULL,
    risk_score REAL,
    risk_bucket TEXT,
    model_version TEXT,
    scored_at TEXT,
    FOREIGN KEY (neo_row_id) REFERENCES near_earth_objects (id)
);
""")

# This index makes the "WHERE id NOT IN (...)" query much faster
create_index_sql = text("""
CREATE INDEX IF NOT EXISTS idx_neo_row_id 
ON neo_risk_scores (neo_row_id);
""")

with engine.begin() as conn:
    conn.execute(create_table_sql)
    conn.execute(create_index_sql)
# --------------------------------------------------------------------

model = joblib.load(MODEL_PATH)

model = joblib.load(MODEL_PATH)

print("✅ Loaded model from", MODEL_PATH)

# ---------- 3) Fetch rows that are not yet scored ----------
# We assume:
#   - near_earth_objects.id is the PK
#   - We don't want to rescore rows that already exist in neo_risk_scores
query = """
SELECT
    id AS neo_row_id,
    absolute_magnitude_h,
    diameter_m,
    velocity_kms,
    miss_distance_km
FROM near_earth_objects
WHERE id NOT IN (
    SELECT neo_row_id FROM neo_risk_scores
)
"""

df = pd.read_sql(query, engine)

if df.empty:
    print("No new NEO rows to score.")
    exit(0)

print(f"Found {len(df)} NEO rows to score.")

feature_cols = [
    "absolute_magnitude_h",
    "diameter_m",
    "velocity_kms",
    "miss_distance_km",
]

X_new = df[feature_cols]

# ---------- 4) Get probabilitie from the model ----------
# Our model is a LogisticRegression in a Pipeline,s
# so we can use predict_proba to get P(hazardous = 1).
proba = model.predict_proba(X_new)[:, 1]  # probability it's hazardous

# ---------- 5) Convert probability -> bucket ----------
def bucket(p):
    """
    Simple thresholds:
      >= 0.8  -> High
      >= 0.4  -> Medium
      else    -> Low
    You can tweak these after you see real numbers.
    """
    if p >= 0.8:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"

df["risk_score"] = proba
df["risk_bucket"] = [bucket(p) for p in proba]

print("\nExample scores:")
print(df[["neo_row_id", "risk_score", "risk_bucket"]].head())

# ---------- 6) Insert scores into neo_risk_scores ----------
now_str = datetime.utcnow().isoformat()

with engine.begin() as conn:
    for _, row in df.iterrows():
        conn.execute(
            text(
                """
                INSERT INTO neo_risk_scores
                    (neo_row_id, risk_score, risk_bucket, model_version, scored_at)
                VALUES
                    (:neo_row_id, :risk_score, :risk_bucket, :model_version, :scored_at)
                """
            ),
            {
                "neo_row_id": int(row["neo_row_id"]),
                "risk_score": float(row["risk_score"]),
                "risk_bucket": row["risk_bucket"],
                "model_version": MODEL_VERSION,
                "scored_at": now_str,
            },
        )

print(f"\n✅ Scoring completed. Inserted {len(df)} rows into neo_risk_scores.")
