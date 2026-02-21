"""
End-to-end NEO pipeline for Railway worker:
1) Fetch latest NEO data from NASA NeoWs
2) Upsert into SQLite raw table
3) Train/retrain XGBoost + Isolation Forest
4) Generate predictions for dashboard
5) Repeat every 2 hours (12 runs/day)
"""

import argparse
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier


NASA_NEO_URL = "https://api.nasa.gov/neo/rest/v1/feed"
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
DB_PATH = os.getenv("DB_PATH", "neo.db")

RAW_TABLE = "near_earth_objects"
PRED_TABLE = "neo_predictions"
RUNS_TABLE = "pipeline_runs"

XGB_MODEL_PATH = "neo_hazard_model_xgb_iso.joblib"
ISO_MODEL_PATH = "neo_isolation_forest_model.joblib"
ISO_SCALER_PATH = "neo_isolation_forest_scaler.joblib"  # reserved for compatibility

FEATURE_COLS = [
    "absolute_magnitude_h",
    "diameter_m",
    "velocity_kms",
    "miss_distance_km",
]

RISK_BINS = [-0.000001, 0.3, 0.7, 1.000001]
RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    fetched_rows: int = 0
    raw_upserted_rows: int = 0
    training_rows: int = 0
    predicted_rows: int = 0
    high_risk_rows: int = 0
    anomaly_rows: int = 0
    status: str = "ok"
    error: str = ""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_columns(conn: sqlite3.Connection, table: str, columns_with_type: dict[str, str]) -> None:
    existing = _table_columns(conn, table)
    for col, col_type in columns_with_type.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RAW_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                name TEXT,
                neo_reference_id TEXT,
                absolute_magnitude_h REAL,
                diameter_min_m REAL,
                diameter_max_m REAL,
                diameter_m REAL,
                velocity_kmh REAL,
                velocity_kms REAL,
                miss_distance_km REAL,
                orbiting_body TEXT,
                hazardous INTEGER,
                timestamp TEXT,
                UNIQUE(neo_reference_id, date)
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {PRED_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                name TEXT,
                neo_reference_id TEXT,
                orbiting_body TEXT,
                diameter_m REAL,
                miss_distance_km REAL,
                velocity_kmh REAL,
                hazardous INTEGER,
                absolute_magnitude_h REAL,
                xgb_risk_prob REAL,
                isolation_anomaly_score REAL,
                is_anomaly INTEGER,
                risk_score REAL,
                risk_label TEXT,
                prediction_time_utc TEXT,
                UNIQUE(neo_reference_id, date)
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_time_utc TEXT,
                fetched_rows INTEGER,
                raw_upserted_rows INTEGER,
                training_rows INTEGER,
                predicted_rows INTEGER,
                high_risk_rows INTEGER,
                anomaly_rows INTEGER,
                status TEXT,
                error TEXT
            )
            """
        )
        _ensure_columns(
            conn,
            RAW_TABLE,
            {
                "date": "TEXT",
                "name": "TEXT",
                "neo_reference_id": "TEXT",
                "absolute_magnitude_h": "REAL",
                "diameter_min_m": "REAL",
                "diameter_max_m": "REAL",
                "diameter_m": "REAL",
                "velocity_kmh": "REAL",
                "velocity_kms": "REAL",
                "miss_distance_km": "REAL",
                "orbiting_body": "TEXT",
                "hazardous": "INTEGER",
                "timestamp": "TEXT",
            },
        )
        _ensure_columns(
            conn,
            PRED_TABLE,
            {
                "date": "TEXT",
                "name": "TEXT",
                "neo_reference_id": "TEXT",
                "orbiting_body": "TEXT",
                "diameter_m": "REAL",
                "miss_distance_km": "REAL",
                "velocity_kmh": "REAL",
                "hazardous": "INTEGER",
                "absolute_magnitude_h": "REAL",
                "xgb_risk_prob": "REAL",
                "isolation_anomaly_score": "REAL",
                "is_anomaly": "INTEGER",
                "risk_score": "REAL",
                "risk_label": "TEXT",
                "prediction_time_utc": "TEXT",
            },
        )
        _ensure_columns(
            conn,
            RUNS_TABLE,
            {
                "run_time_utc": "TEXT",
                "fetched_rows": "INTEGER",
                "raw_upserted_rows": "INTEGER",
                "training_rows": "INTEGER",
                "predicted_rows": "INTEGER",
                "high_risk_rows": "INTEGER",
                "anomaly_rows": "INTEGER",
                "status": "TEXT",
                "error": "TEXT",
            },
        )
        conn.commit()


def fetch_neos(start_date: str, end_date: str) -> list[dict]:
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": NASA_API_KEY,
    }
    resp = requests.get(NASA_NEO_URL, params=params, timeout=45)
    resp.raise_for_status()
    payload = resp.json()

    rows: list[dict] = []
    neo_by_date = payload.get("near_earth_objects", {})

    for date_str, neos in neo_by_date.items():
        for neo in neos:
            cad_list = neo.get("close_approach_data", [])
            if not cad_list:
                continue

            cad = cad_list[0]
            diameter_data = neo.get("estimated_diameter", {}).get("meters", {})
            d_min = float(diameter_data.get("estimated_diameter_min", 0) or 0)
            d_max = float(diameter_data.get("estimated_diameter_max", 0) or 0)
            diameter_m = (d_min + d_max) / 2 if (d_min and d_max) else max(d_min, d_max)

            velocity_kmh = float(cad.get("relative_velocity", {}).get("kilometers_per_hour", 0) or 0)
            velocity_kms = float(cad.get("relative_velocity", {}).get("kilometers_per_second", 0) or 0)
            miss_distance_km = float(cad.get("miss_distance", {}).get("kilometers", 0) or 0)

            rows.append(
                {
                    "date": date_str,
                    "name": neo.get("name", "Unknown"),
                    "neo_reference_id": str(neo.get("neo_reference_id", "")),
                    "absolute_magnitude_h": float(neo.get("absolute_magnitude_h", 0) or 0),
                    "diameter_min_m": d_min,
                    "diameter_max_m": d_max,
                    "diameter_m": float(diameter_m or 0),
                    "velocity_kmh": velocity_kmh,
                    "velocity_kms": velocity_kms,
                    "miss_distance_km": miss_distance_km,
                    "orbiting_body": cad.get("orbiting_body", "Earth"),
                    "hazardous": int(bool(neo.get("is_potentially_hazardous_asteroid", False))),
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )

    return rows


def upsert_raw(rows: Iterable[dict]) -> int:
    rows = list(rows)
    if not rows:
        return 0

    payload = [
        (
            r["date"],
            r["name"],
            r["neo_reference_id"],
            r["absolute_magnitude_h"],
            r["diameter_min_m"],
            r["diameter_max_m"],
            r["diameter_m"],
            r["velocity_kmh"],
            r["velocity_kms"],
            r["miss_distance_km"],
            r["orbiting_body"],
            r["hazardous"],
            r["timestamp"],
        )
        for r in rows
    ]

    with get_connection() as conn:
        update_sql = f"""
            UPDATE {RAW_TABLE}
            SET
                name=?,
                absolute_magnitude_h=?,
                diameter_min_m=?,
                diameter_max_m=?,
                diameter_m=?,
                velocity_kmh=?,
                velocity_kms=?,
                miss_distance_km=?,
                orbiting_body=?,
                hazardous=?,
                timestamp=?
            WHERE neo_reference_id=? AND date=?
        """
        update_payload = [
            (
                r["name"],
                r["absolute_magnitude_h"],
                r["diameter_min_m"],
                r["diameter_max_m"],
                r["diameter_m"],
                r["velocity_kmh"],
                r["velocity_kms"],
                r["miss_distance_km"],
                r["orbiting_body"],
                r["hazardous"],
                r["timestamp"],
                r["neo_reference_id"],
                r["date"],
            )
            for r in rows
        ]
        conn.executemany(update_sql, update_payload)

        insert_sql = f"""
            INSERT INTO {RAW_TABLE} (
                date, name, neo_reference_id, absolute_magnitude_h,
                diameter_min_m, diameter_max_m, diameter_m,
                velocity_kmh, velocity_kms, miss_distance_km,
                orbiting_body, hazardous, timestamp
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM {RAW_TABLE} WHERE neo_reference_id=? AND date=?
            )
        """
        insert_payload = [tuple(p) + (p[2], p[0]) for p in payload]
        conn.executemany(insert_sql, insert_payload)
        conn.commit()
    return len(rows)


def load_training_data() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT
                date, name, neo_reference_id, orbiting_body,
                absolute_magnitude_h, diameter_m, velocity_kmh, velocity_kms,
                miss_distance_km, hazardous
            FROM {RAW_TABLE}
            WHERE absolute_magnitude_h IS NOT NULL
              AND diameter_m IS NOT NULL
              AND velocity_kms IS NOT NULL
              AND miss_distance_km IS NOT NULL
            """,
            conn,
        )
    return df


def train_models(df: pd.DataFrame) -> Tuple[Optional[XGBClassifier], Optional[IsolationForest]]:
    if df.empty:
        logger.warning("No data available for training.")
        return None, None

    x = df[FEATURE_COLS].astype(float).fillna(0.0)
    y = df["hazardous"].astype(int)

    iso = IsolationForest(
        contamination=0.02,
        random_state=42,
        n_estimators=200,
    )
    iso.fit(x)
    joblib.dump(iso, ISO_MODEL_PATH)
    joblib.dump({"feature_cols": FEATURE_COLS}, ISO_SCALER_PATH)

    if y.nunique() < 2 or y.sum() < 20:
        logger.warning("Insufficient hazardous diversity for XGBoost training; skipping XGBoost retrain.")
        return None, iso

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = max(1.0, (n_neg / max(n_pos, 1)))

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    xgb.fit(x, y)
    joblib.dump(xgb, XGB_MODEL_PATH)
    return xgb, iso


def _normalize_anomaly_score(raw_scores: np.ndarray) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=float)
    mn = float(np.min(raw_scores))
    mx = float(np.max(raw_scores))
    if mx - mn <= 1e-12:
        return np.zeros_like(raw_scores, dtype=float)
    return 1.0 - ((raw_scores - mn) / (mx - mn))


def score_predictions(
    df: pd.DataFrame, xgb: Optional[XGBClassifier], iso: Optional[IsolationForest]
) -> pd.DataFrame:
    if df.empty:
        return df

    x = df[FEATURE_COLS].astype(float).fillna(0.0)

    if xgb is not None:
        xgb_prob = xgb.predict_proba(x)[:, 1].astype(float)
    else:
        xgb_prob = (
            (df["diameter_m"] > 140).astype(float) * 0.45
            + (df["velocity_kms"] > 15).astype(float) * 0.25
            + (df["miss_distance_km"] < 7_500_000).astype(float) * 0.30
        ).to_numpy(dtype=float)
        xgb_prob = np.clip(xgb_prob, 0.0, 1.0)

    if iso is not None:
        iso_raw = iso.score_samples(x)
        iso_score = _normalize_anomaly_score(iso_raw)
        is_anomaly = (iso.predict(x) == -1).astype(int)
    else:
        iso_score = np.zeros(len(df), dtype=float)
        is_anomaly = np.zeros(len(df), dtype=int)

    risk_score = np.clip(0.6 * xgb_prob + 0.4 * iso_score, 0.0, 1.0)
    risk_label = pd.cut(risk_score, bins=RISK_BINS, labels=RISK_LABELS).astype(str)

    out = df.copy()
    out["xgb_risk_prob"] = xgb_prob
    out["isolation_anomaly_score"] = iso_score
    out["is_anomaly"] = is_anomaly
    out["risk_score"] = risk_score
    out["risk_label"] = risk_label
    out["prediction_time_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def upsert_predictions(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    payload = [
        (
            r.date,
            r.name,
            str(r.neo_reference_id),
            r.orbiting_body,
            float(r.diameter_m),
            float(r.miss_distance_km),
            float(r.velocity_kmh),
            int(r.hazardous),
            float(r.absolute_magnitude_h),
            float(r.xgb_risk_prob),
            float(r.isolation_anomaly_score),
            int(r.is_anomaly),
            float(r.risk_score),
            str(r.risk_label),
            str(r.prediction_time_utc),
        )
        for r in df.itertuples(index=False)
    ]

    with get_connection() as conn:
        update_sql = f"""
            UPDATE {PRED_TABLE}
            SET
                name=?,
                orbiting_body=?,
                diameter_m=?,
                miss_distance_km=?,
                velocity_kmh=?,
                hazardous=?,
                absolute_magnitude_h=?,
                xgb_risk_prob=?,
                isolation_anomaly_score=?,
                is_anomaly=?,
                risk_score=?,
                risk_label=?,
                prediction_time_utc=?
            WHERE neo_reference_id=? AND date=?
        """
        update_payload = [
            (
                p[1],   # name
                p[3],   # orbiting_body
                p[4],   # diameter_m
                p[5],   # miss_distance_km
                p[6],   # velocity_kmh
                p[7],   # hazardous
                p[8],   # absolute_magnitude_h
                p[9],   # xgb_risk_prob
                p[10],  # isolation_anomaly_score
                p[11],  # is_anomaly
                p[12],  # risk_score
                p[13],  # risk_label
                p[14],  # prediction_time_utc
                p[2],   # neo_reference_id
                p[0],   # date
            )
            for p in payload
        ]
        conn.executemany(update_sql, update_payload)

        insert_sql = f"""
            INSERT INTO {PRED_TABLE} (
                date, name, neo_reference_id, orbiting_body,
                diameter_m, miss_distance_km, velocity_kmh, hazardous,
                absolute_magnitude_h, xgb_risk_prob, isolation_anomaly_score,
                is_anomaly, risk_score, risk_label, prediction_time_utc
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM {PRED_TABLE} WHERE neo_reference_id=? AND date=?
            )
        """
        insert_payload = [tuple(p) + (p[2], p[0]) for p in payload]
        conn.executemany(insert_sql, insert_payload)
        conn.commit()
    return len(payload)


def save_run_stats(stats: RunStats) -> None:
    with get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO {RUNS_TABLE} (
                run_time_utc, fetched_rows, raw_upserted_rows, training_rows,
                predicted_rows, high_risk_rows, anomaly_rows, status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                stats.fetched_rows,
                stats.raw_upserted_rows,
                stats.training_rows,
                stats.predicted_rows,
                stats.high_risk_rows,
                stats.anomaly_rows,
                stats.status,
                stats.error[:1000],
            ),
        )
        conn.commit()


def run_pipeline_once() -> RunStats:
    stats = RunStats()
    try:
        init_db()

        today_utc = datetime.now(timezone.utc).date()
        start_date = (today_utc - timedelta(days=1)).isoformat()
        end_date = (today_utc + timedelta(days=6)).isoformat()

        rows = fetch_neos(start_date, end_date)
        stats.fetched_rows = len(rows)
        stats.raw_upserted_rows = upsert_raw(rows)

        train_df = load_training_data()
        stats.training_rows = len(train_df)

        xgb, iso = train_models(train_df)
        pred_df = score_predictions(train_df, xgb, iso)
        stats.predicted_rows = upsert_predictions(pred_df)
        stats.high_risk_rows = int((pred_df["risk_label"] == "HIGH").sum()) if not pred_df.empty else 0
        stats.anomaly_rows = int(pred_df["is_anomaly"].sum()) if not pred_df.empty else 0
        stats.status = "ok"

        logger.info(
            "Pipeline cycle complete | fetched=%d raw_upserted=%d trained_rows=%d predicted=%d high_risk=%d anomalies=%d",
            stats.fetched_rows,
            stats.raw_upserted_rows,
            stats.training_rows,
            stats.predicted_rows,
            stats.high_risk_rows,
            stats.anomaly_rows,
        )
    except Exception as exc:
        stats.status = "error"
        stats.error = str(exc)
        logger.exception("Pipeline cycle failed: %s", exc)
    finally:
        save_run_stats(stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="NEO end-to-end pipeline")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run one cycle and exit",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.getenv("PIPELINE_INTERVAL_SECONDS", "7200")),
        help="Loop interval in seconds (default: 7200 = 2 hours)",
    )
    args = parser.parse_args()

    if args.run_once:
        run_pipeline_once()
        return

    logger.info("Starting continuous pipeline loop | interval=%ss", args.interval_seconds)
    while True:
        cycle_start = time.time()
        run_pipeline_once()
        elapsed = int(time.time() - cycle_start)
        sleep_for = max(30, args.interval_seconds - elapsed)
        logger.info("Sleeping for %ss before next cycle", sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
