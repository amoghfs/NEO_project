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

# ---------------- DB ----------------
DB_PATH = "neo.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

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

feature_cols = [
    "absolute_magnitude_h",
    "diameter_m",
    "velocity_kms",
    "miss_distance_km",
]

X = df[feature_cols]
y = df["hazardous"].astype(int)

# ---------------- Isolation Forest feature ----------------
iso = IsolationForest(contamination=0.01, random_state=42)
iso_scores = iso.fit_predict(X)
X["iso_anomaly"] = (iso_scores == -1).astype(int)

# ---------------- Train / test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- SMOTE ----------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ---------------- Preprocessing ----------------
def log1p_array(X):
    return np.log1p(X)

log_tf = FunctionTransformer(log1p_array, feature_names_out="one-to-one")

# ---------------- XGBoost ----------------
n_pos = y_train_res.sum()
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
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
            ),
        ),
    ]
)

print("Training model...")
pipe.fit(X_train_res, y_train_res)

# ---------------- Evaluation ----------------
y_proba = pipe.predict_proba(X_test)[:, 1]

# Threshold tuning (PR-optimized)
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = np.argmax(f1)
best_threshold = thresholds[best_idx]

y_pred = (y_proba >= best_threshold).astype(int)

print("\nBest threshold:", round(best_threshold, 3))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("PR  AUC:", average_precision_score(y_test, y_proba))

# ---------------- Save ----------------
joblib.dump(pipe, "neo_xgb_smote_model.joblib")
print("\n✅ Model saved")


    


