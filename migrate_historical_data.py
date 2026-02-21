
#C:\Users\agurl\OneDrive\Desktop\Amogh_Stuff\Synopsys_2025\Github_Folder\NEO_project\neo1.db

import os
import sys
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


# ----------------------------
# Configuration
# ----------------------------
NEW_DB_PATH = "neo.db"
PRED_TABLE = "neo_predictions"

XGBOOST_MODEL_PATH = "neo_hazard_model_xgb_iso.joblib"
ISOLATION_FOREST_MODEL_PATH = "neo_isolation_forest_model.joblib"
SCALER_PATH = "neo_isolation_forest_scaler.joblib"


# ----------------------------
# Helpers
# ----------------------------

def log1p_array(X):
    # X is a numpy array from the imputer
    return np.log1p(X)
def _clean_path(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    s = os.path.expanduser(s)
    return os.path.normpath(s)


def load_historical_data(source_path: str) -> pd.DataFrame | None:
    print(f"Loading historical data from: {source_path}")

    if source_path.lower().endswith(".db"):
        with sqlite3.connect(source_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [r[0] for r in cursor.fetchall()]
            if not tables:
                print("❌ No user tables found in source DB.")
                return None

            print(f"Found tables: {tables}")
            if len(tables) == 1:
                table_name = tables[0]
            else:
                print("Multiple tables found. Choose the NEO table:")
                for i, t in enumerate(tables):
                    print(f"{i+1}. {t}")
                choice = int(input("Enter number: ").strip()) - 1
                if choice < 0 or choice >= len(tables):
                    print("❌ Invalid selection.")
                    return None
                table_name = tables[choice]

            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    elif source_path.lower().endswith(".csv"):
        df = pd.read_csv(source_path)

    elif source_path.lower().endswith(".parquet"):
        df = pd.read_parquet(source_path)

    else:
        print("❌ Unsupported file type.")
        return None

    print(f"Loaded {len(df):,} records")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to the logical names we will use for modeling.
    We keep magnitude as absolute_magnitude_h to match your DB schema.
    """
    column_mapping = {
        # Date
        "close_approach_date": "date",
        "approach_date": "date",

        # ID/name
        "designation": "name",
        "neo_id": "neo_reference_id",
        "reference_id": "neo_reference_id",

        # Physical
        "estimated_diameter_m": "diameter_m",
        "diameter": "diameter_m",
        "est_diameter_meters": "diameter_m",

        # Orbital
        "miss_distance": "miss_distance_km",
        "relative_velocity": "velocity_kmh",
        "velocity": "velocity_kmh",

        # Hazard
        "is_hazardous": "hazardous",
        "potentially_hazardous": "hazardous",
        "is_potentially_hazardous_asteroid": "hazardous",

        # Magnitude
        "absolute_magnitude": "absolute_magnitude_h",
        "h_mag": "absolute_magnitude_h",
        "absolute_magnitude_h": "absolute_magnitude_h",
    }
    df = df.rename(columns=column_mapping)

    # Derive diameter_m if min/max exist and diameter_m missing
    if "diameter_m" not in df.columns and "diameter_min" in df.columns and "diameter_max" in df.columns:
        df["diameter_m"] = (
            pd.to_numeric(df["diameter_min"], errors="coerce") +
            pd.to_numeric(df["diameter_max"], errors="coerce")
        ) / 2.0

    if "orbiting_body" not in df.columns:
        df["orbiting_body"] = "Earth"

    # ensure required modeling cols exist
    for c in ["diameter_m", "miss_distance_km", "velocity_kmh", "absolute_magnitude_h", "hazardous"]:
        if c not in df.columns:
            df[c] = 0

    df["hazardous"] = pd.to_numeric(df["hazardous"], errors="coerce").fillna(0).astype(int)

    # Clean numeric
    for c in ["diameter_m", "miss_distance_km", "velocity_kmh", "absolute_magnitude_h"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def _find_prediction_time_column(table_cols: list[str]) -> str | None:
    """
    Your UI truncates the name; try common possibilities.
    """
    candidates = [
        "prediction_time_utc",
        "prediction_time",
        "prediction_time_utc_text",
        "prediction_time_timestamp",
    ]
    for c in candidates:
        if c in table_cols:
            return c

    # fallback: any column starting with prediction_time
    for c in table_cols:
        if c.lower().startswith("prediction_time"):
            return c

    return None


def add_predictions(df: pd.DataFrame) -> pd.DataFrame:
    print("Loading models and generating predictions...")

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    xgb_model = joblib.load(XGBOOST_MODEL_PATH) if os.path.exists(XGBOOST_MODEL_PATH) else None
    iso_model = joblib.load(ISOLATION_FOREST_MODEL_PATH) if os.path.exists(ISOLATION_FOREST_MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

    # Base 4 features
    base_features = ["diameter_m", "miss_distance_km", "velocity_kmh", "absolute_magnitude_h"]
    X4 = df[base_features].fillna(0).to_numpy()

    # Scale for isolation forest (if scaler exists)
    try:
        X4_scaled = scaler.transform(X4) if scaler is not None else X4
    except Exception as e:
        print(f"⚠️ Scaler failed, using unscaled features: {e}")
        X4_scaled = X4

    # ---------------- XGBoost ----------------
    df["xgb_risk_prob"] = 0.0
    if xgb_model is not None:
        print("Running XGBoost predictions...")
        try:
            # Determine expected feature count for first step in pipeline if possible
            expected = None
            if hasattr(xgb_model, "n_features_in_"):
                expected = int(xgb_model.n_features_in_)

            # Many people save a Pipeline(Imputer+Transformer+Model); the first step may define n_features_in_
            if expected is None and hasattr(xgb_model, "steps"):
                for _, step in xgb_model.steps:
                    if hasattr(step, "n_features_in_"):
                        expected = int(step.n_features_in_)
                        break

            # If it expects 5, pad with hazardous as the 5th feature.
            if expected == 5:
                X5 = np.column_stack([X4, df["hazardous"].fillna(0).to_numpy()])
                df["xgb_risk_prob"] = xgb_model.predict_proba(X5)[:, 1]
            else:
                # Default try 4
                df["xgb_risk_prob"] = xgb_model.predict_proba(X4)[:, 1]
        except Exception as e:
            print(f"⚠️ XGBoost predict_proba failed: {e}")
            df["xgb_risk_prob"] = 0.0
    else:
        print("⚠️ XGBoost model not found, using zeros.")

    # ---------------- Isolation Forest ----------------
    df["anomaly_score"] = 0.0
    df["iso_anomaly"] = 0
    df["anomaly_label"] = 0

    if iso_model is not None:
        print("Running Isolation Forest predictions...")
        try:
            scores = iso_model.score_samples(X4_scaled)
            mn, mx = float(np.min(scores)), float(np.max(scores))
            if mx - mn > 0:
                # 0..1 where higher => more anomalous
                df["anomaly_score"] = 1 - (scores - mn) / (mx - mn)
            else:
                df["anomaly_score"] = 0.0

            preds = (iso_model.predict(X4_scaled) == -1).astype(int)
            df["iso_anomaly"] = preds
            df["anomaly_label"] = preds
        except Exception as e:
            print(f"⚠️ Isolation Forest error: {e}")
    else:
        print("⚠️ Isolation Forest model not found, using zeros.")

    # Combined risk (match your table: risk_score, risk_label)
    df["risk_score"] = (0.6 * df["xgb_risk_prob"] + 0.4 * df["anomaly_score"]).astype(float)
    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[-0.000001, 0.3, 0.7, 1.000001],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)

    # timezone-aware utc timestamp to avoid DeprecationWarning
    df["prediction_time_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print("Predictions completed.")
    return df


def save_to_database(df: pd.DataFrame) -> None:
    """
    UPSERT into existing neo_predictions schema.

    - Auto-detects actual table columns and only writes those.
    - Creates UNIQUE index on (neo_reference_id, date) so ON CONFLICT works reliably.
    """
    print("Saving to database with UPSERT...")

    with sqlite3.connect(NEW_DB_PATH) as conn:
        table_cols = _get_table_columns(conn, PRED_TABLE)
        if not table_cols:
            raise RuntimeError(f"Table {PRED_TABLE} not found in {NEW_DB_PATH}. Create it first.")

        pred_time_col = _find_prediction_time_column(table_cols)

        # Create a unique index for upsert key (neo_reference_id, date)
        # This will fail only if you have duplicates already; if so, we need to clean them once.
        conn.execute(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS ux_{PRED_TABLE}_ref_date
            ON {PRED_TABLE}(neo_reference_id, date)
        """)

        # Build a dataframe that matches your existing schema
        # Map internal columns to DB columns (most are same already)
        candidate_map = {
            "neo_reference_id": "neo_reference_id",
            "date": "date",
            "name": "name",
            "orbiting_body": "orbiting_body",
            "absolute_magnitude_h": "absolute_magnitude_h",
            "diameter_m": "diameter_m",
            "miss_distance_km": "miss_distance_km",
            "velocity_kmh": "velocity_kmh",
            "hazardous": "hazardous",
            "anomaly_score": "anomaly_score",
            "anomaly_label": "anomaly_label",
            "iso_anomaly": "iso_anomaly",
            "risk_score": "risk_score",
            "risk_label": "risk_label",
        }

        if pred_time_col:
            candidate_map["prediction_time_utc"] = pred_time_col

        # Keep only columns that exist in DB and exist in df
        db_write_cols = []
        for df_col, db_col in candidate_map.items():
            if db_col in table_cols and df_col in df.columns:
                db_write_cols.append((df_col, db_col))

        if not db_write_cols:
            raise RuntimeError("No writable columns matched between dataframe and neo_predictions table.")

        df_write = pd.DataFrame()
        for df_col, db_col in db_write_cols:
            df_write[db_col] = df[df_col]

        # Important: do NOT try to write pred_id (PK) unless you manage it
        # Optional: if source_id exists in DB, set it to 1 (or 0) if you want
        if "source_id" in table_cols and "source_id" not in df_write.columns:
            df_write["source_id"] = 1

        # Ensure key columns exist
        if "neo_reference_id" not in df_write.columns or "date" not in df_write.columns:
            raise RuntimeError("neo_reference_id and date are required for UPSERT key.")

        # Convert types
        df_write["neo_reference_id"] = df_write["neo_reference_id"].astype(str)
        df_write["date"] = df_write["date"].astype(str)

        # Build UPSERT SQL
        cols = df_write.columns.tolist()
        placeholders = ",".join(["?"] * len(cols))
        col_list = ",".join(cols)

        update_cols = [c for c in cols if c not in ("neo_reference_id", "date")]
        update_set = ",".join([f"{c}=excluded.{c}" for c in update_cols])

        sql = f"""
            INSERT INTO {PRED_TABLE} ({col_list})
            VALUES ({placeholders})
            ON CONFLICT(neo_reference_id, date) DO UPDATE SET
            {update_set}
        """

        conn.executemany(sql, df_write.itertuples(index=False, name=None))
        conn.commit()

        final_count = conn.execute(f"SELECT COUNT(*) FROM {PRED_TABLE}").fetchone()[0]

    print(f"✅ Upserted {len(df_write):,} rows. {PRED_TABLE} now has {final_count:,} records.")


def main() -> int:
    print("=" * 60)
    print("NEO Historical Data Migration")
    print("=" * 60)

    if len(sys.argv) > 1:
        source_path = _clean_path(sys.argv[1])
    else:
        source_path = _clean_path(input("Enter path to historical data file (DB/CSV): "))

    if not os.path.exists(source_path):
        print(f"❌ File not found: {source_path}")
        return 1

    df = load_historical_data(source_path)
    if df is None or df.empty:
        print("❌ No data loaded.")
        return 1

    df = standardize_columns(df)

    print("\nData preview:")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")

    confirm = input("\nDoes this look correct? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return 1

    df = add_predictions(df)
    save_to_database(df)

    print("\n✅ Migration complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())