import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = os.getenv("DB_PATH", "neo.db")
PRED_TABLE = "neo_predictions"
RUNS_TABLE = "pipeline_runs"
CACHE_TTL_SECONDS = int(os.getenv("APP_CACHE_TTL_SECONDS", "60"))

st.set_page_config(page_title="NEO Risk Dashboard", layout="wide")
st.title("Near-Earth Object Risk Dashboard")


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_predictions() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {PRED_TABLE}", conn)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_latest_run() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM {RUNS_TABLE} ORDER BY run_time_utc DESC LIMIT 1",
            conn,
        )


def format_utc(ts: str) -> str:
    if not ts:
        return "N/A"
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return ts


try:
    df = load_predictions()
    latest_run = load_latest_run()
except Exception as exc:
    st.error(f"Unable to load dashboard data from {DB_PATH}: {exc}")
    st.stop()

if latest_run.empty:
    st.info("The hourly pipeline has not written any run history yet.")
else:
    run = latest_run.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Last pipeline run", format_utc(run.get("run_time_utc", "")))
    cols[1].metric("Run status", str(run.get("status", "unknown")).upper())
    cols[2].metric("Fetched rows", int(run.get("fetched_rows", 0) or 0))
    cols[3].metric("Predictions updated", int(run.get("predicted_rows", 0) or 0))
    if run.get("error"):
        st.warning(f"Latest pipeline error: {run['error']}")

if df.empty:
    st.warning(
        f"No rows found in '{PRED_TABLE}'. Start the Railway worker or run `python neo_pipeline.py --run-once` first."
    )
    st.stop()

# Cleanup and derived fields
df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
df = df.dropna(subset=["risk_score"])

if "date" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df["date_parsed"] = pd.NaT

st.sidebar.header("Filters")
min_score = float(df["risk_score"].min())
max_score = float(df["risk_score"].max())
score_range = st.sidebar.slider("Risk score range", min_score, max_score, (min_score, max_score))
show_only_high = st.sidebar.checkbox("Show only HIGH risk", value=False)

if "prediction_time_utc" in df.columns:
    prediction_times = sorted(x for x in df["prediction_time_utc"].dropna().unique().tolist() if x)
    if prediction_times:
        selected_prediction_time = st.sidebar.selectbox(
            "Prediction batch",
            options=prediction_times,
            index=len(prediction_times) - 1,
            format_func=format_utc,
        )
        df = df[df["prediction_time_utc"] == selected_prediction_time].copy()

df_f = df[(df["risk_score"] >= score_range[0]) & (df["risk_score"] <= score_range[1])].copy()
if show_only_high and "risk_label" in df_f.columns:
    df_f = df_f[df_f["risk_label"] == "HIGH"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions", len(df_f))
c2.metric("High risk", int((df_f["risk_label"] == "HIGH").sum()) if "risk_label" in df_f.columns else 0)
c3.metric("Avg score", round(float(df_f["risk_score"].mean()), 4) if len(df_f) else 0.0)
c4.metric("Max score", round(float(df_f["risk_score"].max()), 4) if len(df_f) else 0.0)

st.subheader("Risk score distribution")
fig_hist = px.histogram(df_f, x="risk_score", nbins=30)
st.plotly_chart(fig_hist, use_container_width=True)

if df_f["date_parsed"].notna().any():
    tmp = df_f.dropna(subset=["date_parsed"]).sort_values("date_parsed")
    st.subheader("Risk score over date")
    fig_line = px.line(
        tmp,
        x="date_parsed",
        y="risk_score",
        hover_data=["name", "neo_reference_id", "risk_label"] if "risk_label" in tmp.columns else ["name", "neo_reference_id"],
    )
    st.plotly_chart(fig_line, use_container_width=True)

if "miss_distance_km" in df_f.columns and "diameter_m" in df_f.columns:
    st.subheader("Miss distance vs Diameter")
    fig_scatter = px.scatter(
        df_f,
        x="miss_distance_km",
        y="diameter_m",
        color="risk_score",
        hover_data=["name", "date", "risk_label", "neo_reference_id"] if "risk_label" in df_f.columns else ["name", "date", "neo_reference_id"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Top risky objects")
cols = [
    c
    for c in [
        "date",
        "name",
        "neo_reference_id",
        "orbiting_body",
        "diameter_m",
        "miss_distance_km",
        "velocity_kmh",
        "risk_score",
        "risk_label",
        "hazardous",
        "prediction_time_utc",
    ]
    if c in df_f.columns
]
st.dataframe(df_f.sort_values("risk_score", ascending=False)[cols].head(100), use_container_width=True)

st.caption("Railway web reads the latest predictions from SQLite. The worker refreshes the data and retrains the models every hour.")
