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
    precision_recall_curve,
)

from imblearn.over_sampling import SMOTE
import joblib

# ---------- 1) DB connection ----------
DB_PATH = "neo.db"
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

# Drop rows missing all features
df = df.dropna(
    subset=["absolute_magnitude_h", "diameter_m", "velocity_kms", "miss_distance_km"],
    how="all",
)

X = df[
    ["absolute_magnitude_h", "diameter_m", "velocity_kms", "miss_distance_km"]
]
y = df["hazardous"].astype(int)

# ---------- 3) Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# ---------- 4) SMOTE (TRAINING DATA ONLY) ----------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE: ", np.bincount(y_train_res))

# ---------- 5) Pipeline ----------
def log1p_array(X):
    return np.log1p(X)

log_tf = FunctionTransformer(log1p_array, feature_names_out="one-to-one")

pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("log", log_tf),
        ("scale", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
            ),
        ),
    ]
)

print("\nTraining Logistic Regression with SMOTE...")
pipe.fit(X_train_res, y_train_res)

# ---------- 6) Evaluation ----------
y_proba = pipe.predict_proba(X_test)[:, 1]

# Threshold tuning for PR
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
best_threshold = thresholds[f1.argmax()]

print(f"Best threshold: {best_threshold:.3f}")

y_pred = (y_proba >= best_threshold).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("PR  AUC:", average_precision_score(y_test, y_proba))

# ---------- 7) Save model ----------
MODEL_PATH = "neo_hazard_model_lr_smote.joblib"
joblib.dump(pipe, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
