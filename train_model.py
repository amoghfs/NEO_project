import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import joblib

# ---------- 1) DB connection (SQLite) ----------
DB_PATH = "neo.db"  # make sure this matches your import script
engine = create_engine(f"sqlite:///{DB_PATH}")

# ---------- 2) Load data ----------
query = """
SELECT
    absolute_magnitude_h,
    diameter_m,  
    velocity_kms,
    miss_distance_km,
    hazardous
FROM near_earth_objects
"""
df = pd.read_sql(query, engine)

print("Loaded data:", df.shape)
print(df.head())

# Drop rows missing all features
df = df.dropna(
    subset=["absolute_magnitude_h", "diameter_m", "velocity_kms", "miss_distance_km"],
    how="all",
)

feature_cols = [
    "absolute_magnitude_h",
    "diameter_m",
    "velocity_kms",
    "miss_distance_km",
]

X = df[feature_cols]
y = df["hazardous"].astype(int)

# ---------- 3) Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ---------- 4) Pipeline ----------
def log1p_array(X):
    return np.log1p(X)

log_tf = FunctionTransformer(log1p_array, feature_names_out="one-to-one")

pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("log", log_tf),                       # ← FIXED: comma added
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
            ),
        ),
    ]
)

print("\nTraining model...")
pipe.fit(X_train, y_train)

# ---------- 5) Evaluation ----------
y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("PR  AUC:", average_precision_score(y_test, y_proba))

# ---------- 6) Save model ----------  
MODEL_PATH = "neo_hazard_model.joblib"
joblib.dump(pipe, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")


