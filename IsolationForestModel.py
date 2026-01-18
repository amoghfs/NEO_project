import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# USER CONFIG
DB_PATH = "neo.db"
MODEL_OUTPATH = "isolation_forest_model.joblib"
TABLE_NAME = "near_earth_objects"
TARGET_COL = None
RANDOM_STATE = 42
CONTAMINATION = 0.02


# Loads a table from SQLite DB into a pandas DataFrame
def load_from_database():
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT * FROM {TABLE_NAME};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"Loaded {len(df)} rows from {DB_PATH}:{TABLE_NAME}")
    return df


def build_preprocessor(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                Pipeline([
                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False))
                ]),
                cat_cols
            )
        ],
        remainder="drop"
    )

    return preprocessor


def train_isolation_forest(X):
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )
    model.fit(X)
    return model


# Converts scikit -1/1 output into 1=anomaly, 0=normal
def predict(model, X):
    raw = model.predict(X)
    return np.where(raw == -1, 1, 0)


def visualize(X, preds):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(7, 5))
    normal = preds == 0
    anomaly = preds == 1

    plt.scatter(X2[normal, 0], X2[normal, 1], s=10, alpha=0.6, label="normal")
    plt.scatter(X2[anomaly, 0], X2[anomaly, 1], s=30, marker="x", label="anomaly")

    plt.title("PCA Visualization of NEO Data (Isolation Forest)")
    plt.legend()
    plt.show()


def main():
    # LOADING DATA
    df = load_from_database()

    # Separate labels if they exist
    if TARGET_COL and TARGET_COL in df.columns:
        labels = df[TARGET_COL]
        df = df.drop(columns=[TARGET_COL])
    else:
        labels = None

    # PREPROCESS
    preprocessor = build_preprocessor(df)
    X = preprocessor.fit_transform(df)

    # TRAIN MODEL
    model = train_isolation_forest(X)

    # PREDICT
    preds = predict(model, X)
    print(f"Detected {preds.sum()} anomalies out of {len(preds)} rows.")

    # SAVE MODEL
    joblib.dump({"model": model, "preprocessor": preprocessor}, MODEL_OUTPATH)
    print(f"Saved model to {MODEL_OUTPATH}")

    # -------------------------------------------------------------
    # SAVE RESULTS TO DATABASE
    # -------------------------------------------------------------
    conn = sqlite3.connect(DB_PATH)

    # Isolation Forest anomaly score (higher = more anomalous)
    scores = model.decision_function(X)

    # Create output DataFrame with all original columns
    out_df = df.copy()
    out_df["anomaly_score"] = scores
    out_df["anomaly_label"] = preds

    # Ensure an ID column exists
    if "id" not in out_df.columns:
        out_df.insert(0, "id", np.arange(len(out_df)))

    # Save table (replaces if already exists)
    out_df.to_sql("isolation_forest_scores", conn, if_exists="replace", index=False)
    conn.close()

    print("Saved anomaly results → isolation_forest_scores table in neo.db")

    # VISUALIZE
    visualize(X, preds)


if __name__ == "__main__":
    main()
