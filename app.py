import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DB_PATH = "neo.db"   # <-- change to "neo.db" locally
PRED_TABLE = "neo_predictions"

st.set_page_config(page_title="NEO Risk Dashboard", layout="wide")
st.title("Near-Earth Object Risk Dashboard")

@st.cache_data(ttl=5)
def load_predictions():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {PRED_TABLE}", conn)
    return df

df = load_predictions()

if df.empty:
    st.warning(f"No rows found in '{PRED_TABLE}'. Run XGBOOST_model3.py first to generate predictions.")
    st.stop()

# cleanup
df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
df = df.dropna(subset=["risk_score"])

# parse date if present
if "date" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df["date_parsed"] = pd.NaT

# Sidebar controls
st.sidebar.header("Filters")
min_score = float(df["risk_score"].min())
max_score = float(df["risk_score"].max())
score_range = st.sidebar.slider("Risk score range", min_score, max_score, (min_score, max_score))

show_only_high = st.sidebar.checkbox("Show only HIGH risk", value=False)

df_f = df[(df["risk_score"] >= score_range[0]) & (df["risk_score"] <= score_range[1])].copy()
if show_only_high and "risk_label" in df_f.columns:
    df_f = df_f[df_f["risk_label"] == "HIGH"]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions", len(df_f))
c2.metric("High risk", int((df_f["risk_label"] == "HIGH").sum()) if "risk_label" in df_f.columns else 0)
c3.metric("Avg score", float(df_f["risk_score"].mean()) if len(df_f) else 0.0)
c4.metric("Max score", float(df_f["risk_score"].max()) if len(df_f) else 0.0)

# Charts
st.subheader("Risk score distribution")
fig_hist = px.histogram(df_f, x="risk_score", nbins=30)
st.plotly_chart(fig_hist, use_container_width=True)

# Risk over time (date)
if df_f["date_parsed"].notna().any():
    tmp = df_f.dropna(subset=["date_parsed"]).sort_values("date_parsed")
    st.subheader("Risk score over date")
    fig_line = px.line(
        tmp,
        x="date_parsed",
        y="risk_score",
        hover_data=["name", "neo_reference_id", "risk_label"] if "risk_label" in tmp.columns else ["name", "neo_reference_id"]
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Scatter: miss distance vs diameter
if "miss_distance_km" in df_f.columns and "diameter_m" in df_f.columns:
    st.subheader("Miss distance vs Diameter (colored by risk)")
    fig_scatter = px.scatter(
        df_f,
        x="miss_distance_km",
        y="diameter_m",
        color="risk_score",
        hover_data=["name", "date", "risk_label", "neo_reference_id"] if "risk_label" in df_f.columns else ["name", "date", "neo_reference_id"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Table
st.subheader("Top risky objects")
cols = [c for c in ["date", "name", "neo_reference_id", "orbiting_body", "diameter_m", "miss_distance_km", "velocity_kmh", "risk_score", "risk_label", "hazardous", "prediction_time_utc"] if c in df_f.columns]
st.dataframe(df_f.sort_values("risk_score", ascending=False)[cols].head(100), use_container_width=True)

st.caption("Run XGBOOST_model3.py again to append a new batch into neo_predictions, then refresh this page.")
