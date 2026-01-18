import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import joblib

# ---------- 1) DB connection (SQLite) ----------
DB_PATH = "neo.db"  # make sure this matches your import script
engine = create_engine(f"sqlite:///{DB_PATH}")

# ---------- 2) Load data from near_earth_objects ----------
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

# Drop rows where all features are missing (safety)
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
y = df["hazardous"].astype(int)  # 0/1 label

# ---------- 3) Optional: add Isolation Forest anomaly score ----------
iso = IsolationForest(contamination=0.01, random_state=42)
iso_scores = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
X['iso_score'] = (iso_scores == -1).astype(int)  # 1 if anomaly, 0 if normal

# ---------- 4) Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # keeps class ratio similar
)

# ---------- 5) Handle class imbalance with SMOTE ----------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# ---------- 6) Build ML pipeline ----------
def log1p_array(X):
    # X is a numpy array from the imputer
    return np.log1p(X)

log_tf = FunctionTransformer(log1p_array, feature_names_out="one-to-one")

# Compute scale_pos_weight for XGBoost
n_pos = sum(y_train_res)
n_neg = len(y_train_res) - n_pos
scale_pos_weight = n_neg / n_pos

pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("log", log_tf),
        ("scale", StandardScaler()),
        (
            "clf",
            XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
            ),
        ),
    ]
)

print("\nTraining model...")
pipe.fit(X_train_res, y_train_res)

# ---------- 7) Evaluate ----------
y_proba = pipe.predict_proba(X_test)[:, 1]  # probability of hazardous

# ---------- Automatic threshold tuning ----------
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
print(f"Best threshold for max F1 / PR: {best_threshold:.3f}")

y_pred = (y_proba >= best_threshold).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("PR  AUC:", average_precision_score(y_test, y_proba))

# ---------- 8) Save trained model ----------
MODEL_PATH = "neo_hazard_model_xgb_iso.joblib"
joblib.dump(pipe, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")




