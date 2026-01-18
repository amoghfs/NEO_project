# import numpy as np
# import pandas as pd
# from sqlalchemy import create_engine

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import FunctionTransformer, StandardScaler
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     average_precision_score,
#     precision_recall_curve,
# )

# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
# from sklearn.ensemble import IsolationForest
# import joblib

# # ---------------- DB ----------------
# DB_PATH = "neo.db"
# engine = create_engine(f"sqlite:///{DB_PATH}")

# query = """
# SELECT
#     absolute_magnitude_h,
#     diameter_m,
#     velocity_kms,
#     miss_distance_km,
#     hazardous
# FROM near_earth_objects
# """
# df = pd.read_sql(query, engine)

# feature_cols = [
#     "absolute_magnitude_h",
#     "diameter_m",
#     "velocity_kms",
#     "miss_distance_km",
# ]

# X = df[feature_cols]
# y = df["hazardous"].astype(int)

# # ---------------- Isolation Forest feature ----------------
# iso = IsolationForest(contamination=0.01, random_state=42)
# iso_scores = iso.fit_predict(X)
# X["iso_anomaly"] = (iso_scores == -1).astype(int)

# # ---------------- Train / test split ----------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # ---------------- SMOTE ----------------
# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # ---------------- Preprocessing ----------------
# def log1p_array(X):
#     return np.log1p(X)

# log_tf = FunctionTransformer(log1p_array, feature_names_out="one-to-one")

# # ---------------- XGBoost ----------------
# n_pos = y_train_res.sum()
# n_neg = len(y_train_res) - n_pos
# scale_pos_weight = n_neg / n_pos

# pipe = Pipeline(
#     steps=[
#         ("impute", SimpleImputer(strategy="median")),
#         ("log", log_tf),
#         ("scale", StandardScaler()),
#         (
#             "clf",
#             XGBClassifier(
#                 n_estimators=400,
#                 max_depth=5,
#                 learning_rate=0.05,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 objective="binary:logistic",
#                 eval_metric="logloss",
#                 scale_pos_weight=scale_pos_weight,
#                 random_state=42,
#             ),
#         ),
#     ]
# )

# print("Training model...")
# pipe.fit(X_train_res, y_train_res)

# # ---------------- Evaluation ----------------
# y_proba = pipe.predict_proba(X_test)[:, 1]

# # Threshold tuning (PR-optimized)
# precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# f1 = 2 * precision * recall / (precision + recall + 1e-9)
# best_idx = np.argmax(f1)
# best_threshold = thresholds[best_idx]

# y_pred = (y_proba >= best_threshold).astype(int)

# print("\nBest threshold:", round(best_threshold, 3))
# print("\nConfusion matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification report:")
# print(classification_report(y_test, y_pred, digits=3))

# print("ROC AUC:", roc_auc_score(y_test, y_proba))
# print("PR  AUC:", average_precision_score(y_test, y_proba))

# # ---------------- Save ----------------
# joblib.dump(pipe, "neo_xgb_smote_model.joblib")
# print("\n✅ Model saved")
    


import sqlite3
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone

# =========================================================
# CONFIG (UPDATE THESE TWO PATHS ON YOUR WINDOWS MACHINE)
# =========================================================
DB_PATH = r"neo.db"
MODEL_PATH = r"neo_xgb_smote_model.joblib"

SOURCE_TABLE = "near_earth_objects"
PRED_TABLE = "neo_predictions"

RISK_THRESHOLD = 0.70

# =========================================================
# REQUIRED FOR YOUR JOBLIB UNPICKLE (your earlier error)
# "Can't get attribute 'log1p_array' on <module '__main__' ...>"
# =========================================================
def log1p_array(x):
    return np.log1p(x)

# =========================================================
# OPTION A: Add missing features in-memory (NO model change)
# =========================================================
def add_missing_features_for_model(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """
    Ensures the dataframe contains features expected by the trained pipeline/model,
    WITHOUT changing the model logic. Only adds derived/alias columns in-memory.

    Current supported fix:
      - iso_anomaly derived from anomaly_label (or anomaly_score fallback)
    """
    # Add iso_anomaly if the model expects it and it's not present in DB
    if "iso_anomaly" in required_features and "iso_anomaly" not in df.columns:
        if "anomaly_label" in df.columns:
            vals = pd.Series(df["anomaly_label"])
            uniq = set(vals.dropna().unique().tolist())

            # Common IsolationForest labels: -1 anomaly, 1 normal
            if uniq.issubset({-1, 1}):
                df["iso_anomaly"] = (df["anomaly_label"] == -1).astype(int)
            else:
                # If anomaly_label already 0/1 (or similar)
                df["iso_anomaly"] = vals.fillna(0).astype(int)

        elif "anomaly_score" in df.columns:
            # Fallback: treat top 5% anomaly_score as anomalies
            score = pd.to_numeric(df["anomaly_score"], errors="coerce")
            thr = score.quantile(0.95)
            df["iso_anomaly"] = (score >= thr).fillna(False).astype(int)

        else:
            # Last resort: if nothing exists, default to 0
            df["iso_anomaly"] = 0

    return df


def ensure_prediction_table(conn: sqlite3.Connection):
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
            pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            neo_reference_id TEXT,
            date TEXT,
            name TEXT,
            orbiting_body TEXT,

            absolute_magnitude_h REAL,
            diameter_m REAL,
            miss_distance_km REAL,
            velocity_kmh REAL,

            anomaly_score REAL,
            anomaly_label INTEGER,
            iso_anomaly INTEGER,

            hazardous INTEGER,

            risk_score REAL,
            risk_label TEXT,

            prediction_time_utc TEXT
        )
    """)
    conn.commit()


def main():
    # 1) Load model (pipeline / estimator) - model logic unchanged
    model = joblib.load(MODEL_PATH)

    # 2) Read data from SQLite
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {SOURCE_TABLE}", conn)

    if df.empty:
        raise RuntimeError(f"No rows found in table '{SOURCE_TABLE}'.")

    # 3) Determine feature columns expected by the model
    #    This does NOT change your model logic; it uses what the model expects.
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        raise RuntimeError(
            "Model does not expose feature_names_in_. "
            "Please set feature_cols manually to match the training features."
        )

    # 4) Option A: add missing features in-memory so DB matches model schema
    df = add_missing_features_for_model(df, feature_cols)

    # 5) Validate required features exist now
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required feature columns in DB even after mapping: {missing}")

    # 6) Build X exactly as model expects (no change to model logic)
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # 7) Predict
    if hasattr(model, "predict_proba"):
        risk_score = model.predict_proba(X)[:, 1]
    else:
        risk_score = model.predict(X)

    risk_score = np.asarray(risk_score, dtype=float)
    risk_label = np.where(risk_score >= RISK_THRESHOLD, "HIGH", "LOW")

    pred_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 8) Prepare output (store useful fields for UI)
    #    We include iso_anomaly if present (and it will be, if the model expected it).
    out = pd.DataFrame({
        "source_id": pd.to_numeric(df.get("id"), errors="coerce"),
        "neo_reference_id": df.get("neo_reference_id"),
        "date": df.get("date"),
        "name": df.get("name"),
        "orbiting_body": df.get("orbiting_body"),

        "absolute_magnitude_h": pd.to_numeric(df.get("absolute_magnitude_h"), errors="coerce"),
        "diameter_m": pd.to_numeric(df.get("diameter_m"), errors="coerce"),
        "miss_distance_km": pd.to_numeric(df.get("miss_distance_km"), errors="coerce"),
        "velocity_kmh": pd.to_numeric(df.get("velocity_kmh"), errors="coerce"),

        "anomaly_score": pd.to_numeric(df.get("anomaly_score"), errors="coerce") if "anomaly_score" in df.columns else None,
        "anomaly_label": pd.to_numeric(df.get("anomaly_label"), errors="coerce") if "anomaly_label" in df.columns else None,
        "iso_anomaly": pd.to_numeric(df.get("iso_anomaly"), errors="coerce") if "iso_anomaly" in df.columns else None,

        "hazardous": pd.to_numeric(df.get("hazardous"), errors="coerce") if "hazardous" in df.columns else None,

        "risk_score": risk_score,
        "risk_label": risk_label,
        "prediction_time_utc": pred_time
    })

    # 9) Write to SQLite
    with sqlite3.connect(DB_PATH) as conn:
        ensure_prediction_table(conn)

        # If you want "latest-only" predictions, uncomment:
        # conn.execute(f"DELETE FROM {PRED_TABLE}")
        # conn.commit()

        out.to_sql(PRED_TABLE, conn, if_exists="append", index=False)

    print(f"✅ Saved {len(out)} predictions into '{DB_PATH}' table '{PRED_TABLE}' at {pred_time}")
    print(f"✅ Features used ({len(feature_cols)}): {feature_cols}")



if __name__ == "__main__":
    main()
