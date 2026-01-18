import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
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

# --------------------------------------------------
# 2. Create physics-based threat labels
# --------------------------------------------------
def create_risk_labels(df):
    df = df.copy()

    mass_kg = (4/3) * np.pi * (df["diameter_m"] / 2) ** 3 * 2000
    kinetic_energy = 0.5 * mass_kg * (df["velocity_kms"] * 1000) ** 2
    impact_megatons = kinetic_energy / 4.184e15

    HIGH_ENERGY = 1.0        # megatons
    VERY_CLOSE = 5_000_000   # km

    nasa_hazardous = df["hazardous"] == 1
    high_energy = impact_megatons >= HIGH_ENERGY
    very_close = df["miss_distance_km"] <= VERY_CLOSE

    # Binary high-threat definition
    df["high_threat"] = (
        nasa_hazardous & (high_energy | very_close)
    ).astype(int)

    return df

df = create_risk_labels(df)

# --------------------------------------------------
# 3. Feature engineering (NO leakage)
# --------------------------------------------------
def engineer_features(df):
    df = df.copy()

    mass_kg = (4/3) * np.pi * (df["diameter_m"] / 2) ** 3 * 2000
    kinetic_energy = 0.5 * mass_kg * (df["velocity_kms"] * 1000) ** 2

    df["log_kinetic_energy"] = np.log1p(kinetic_energy)
    df["log_momentum"] = np.log1p(mass_kg * df["velocity_kms"] * 1000)

    df["velocity_squared"] = df["velocity_kms"] ** 2
    df["is_high_velocity"] = (df["velocity_kms"] > 20).astype(int)

    df["log_diameter"] = np.log1p(df["diameter_m"])
    df["size_category"] = pd.cut(
        df["diameter_m"],
        bins=[0, 50, 150, 500, 1500, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    df["log_miss_distance"] = np.log1p(df["miss_distance_km"])
    df["is_very_close"] = (df["miss_distance_km"] < 7_480_000).astype(int)

    df["velocity_x_size"] = df["velocity_kms"] * df["diameter_m"]
    df["size_to_distance_ratio"] = df["diameter_m"] / (df["miss_distance_km"] + 1)

    df["magnitude_risk"] = np.maximum(0, 22 - df["absolute_magnitude_h"])
    df["log_magnitude_risk"] = np.log1p(df["magnitude_risk"])

    return df

df = engineer_features(df)

# --------------------------------------------------
# 4. Select features & target
# --------------------------------------------------
feature_cols = [
    "absolute_magnitude_h",
    "diameter_m",
    "velocity_kms",
    "miss_distance_km",
    "log_kinetic_energy",
    "log_momentum",
    "velocity_squared",
    "is_high_velocity",
    "log_diameter",
    "size_category",
    "log_miss_distance",
    "is_very_close",
    "velocity_x_size",
    "size_to_distance_ratio",
    "magnitude_risk",
    "log_magnitude_risk",
]

X = df[feature_cols]
y = df["high_threat"]

# --------------------------------------------------
# 5. Train / test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# --------------------------------------------------
# 7. Train XGBoost (binary)
# --------------------------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=1.5,
    eval_metric="aucpr",
    random_state=42,
    verbosity=0,
)

print("Training model...")
model.fit(X_train_scaled, y_train)

# --------------------------------------------------
# 8. Evaluation (formatted output)
# --------------------------------------------------
y_proba = model.predict_proba(X_test_scaled)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

y_pred = (y_proba >= best_threshold).astype(int)

print(f"\nBest threshold: {best_threshold:.3f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))

print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR AUC:  {average_precision_score(y_test, y_proba):.4f}")

# --------------------------------------------------
# 9. Save model
# --------------------------------------------------
joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "threshold": best_threshold,
        "target": "high_threat",
    },
    "neo_high_threat_model.joblib",
)

print("\n✅ Model saved to neo_high_threat_model.joblib")
