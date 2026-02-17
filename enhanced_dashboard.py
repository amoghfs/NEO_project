"""
Enhanced NEO Risk Dashboard with Real-time Updates
Displays predictions from both XGBoost and Isolation Forest models
"""

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configuration
DB_PATH = "neo.db"
PRED_TABLE = "neo_predictions"
AUTO_REFRESH_SECONDS = 30  # Auto-refresh every 30 seconds

# Page config
st.set_page_config(
    page_title="NEO Risk Dashboard",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-metric {
        font-size: 24px;
        font-weight: bold;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .risk-low {
        color: #44ff44;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("☄️ Near-Earth Object Risk Assessment Dashboard")
st.markdown("""
**Real-time monitoring and risk assessment of Near-Earth Objects using Machine Learning**

This dashboard combines **XGBoost classification** and **Isolation Forest anomaly detection** 
to provide comprehensive risk assessments of asteroids approaching Earth.
""")

# Cache data with TTL for auto-refresh
@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def load_predictions():
    """Load predictions from database"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {PRED_TABLE}", conn)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=AUTO_REFRESH_SECONDS)
def get_database_stats():
    """Get database statistics"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            stats = {}
            cursor = conn.execute(f"SELECT COUNT(*) FROM {PRED_TABLE}")
            stats['total_records'] = cursor.fetchone()[0]
            
            cursor = conn.execute(f"SELECT MAX(prediction_time_utc) FROM {PRED_TABLE}")
            stats['last_update'] = cursor.fetchone()[0]
            
            return stats
    except Exception as e:
        return {'total_records': 0, 'last_update': None}

# Load data
df = load_predictions()
db_stats = get_database_stats()

# Display last update time
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if db_stats['last_update']:
        last_update = datetime.fromisoformat(db_stats['last_update'])
        st.caption(f"📊 Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        st.caption("📊 No data available yet")

with col2:
    st.caption(f"🗃️ Total records: {db_stats['total_records']:,}")

with col3:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# Check if data exists
if df.empty:
    st.warning(f"""
    **No predictions found in the database.**
    
    To populate the dashboard:
    1. Ensure your XGBoost and Isolation Forest models are saved
    2. Run `realtime_neo_updater.py` to start fetching and processing NEO data
    3. The dashboard will automatically update every {AUTO_REFRESH_SECONDS} seconds
    """)
    st.stop()

# Data cleaning
df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
df["xgb_risk_prob"] = pd.to_numeric(df.get("xgb_risk_prob", 0), errors="coerce")
df["isolation_anomaly_score"] = pd.to_numeric(df.get("isolation_anomaly_score", 0), errors="coerce")
df = df.dropna(subset=["risk_score"])

# Parse dates
if "date" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df["date_parsed"] = pd.NaT

# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Risk score range
min_score = float(df["risk_score"].min())
max_score = float(df["risk_score"].max())
score_range = st.sidebar.slider(
    "Risk Score Range", 
    min_score, 
    max_score, 
    (min_score, max_score),
    help="Filter NEOs by their combined risk score"
)

# Risk level filter
risk_levels = st.sidebar.multiselect(
    "Risk Level",
    options=["LOW", "MEDIUM", "HIGH"],
    default=["LOW", "MEDIUM", "HIGH"],
    help="Filter by risk classification"
)

# Anomaly filter
show_anomalies_only = st.sidebar.checkbox(
    "Show Anomalies Only",
    value=False,
    help="Display only NEOs flagged as anomalies by Isolation Forest"
)

# Date range filter
if df["date_parsed"].notna().any():
    min_date = df["date_parsed"].min().date()
    max_date = df["date_parsed"].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter by approach date"
    )

# Apply filters
df_filtered = df[
    (df["risk_score"] >= score_range[0]) & 
    (df["risk_score"] <= score_range[1])
].copy()

if "risk_label" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["risk_label"].isin(risk_levels)]

if show_anomalies_only and "is_anomaly" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["is_anomaly"] == 1]

if df["date_parsed"].notna().any() and len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered["date_parsed"].dt.date >= date_range[0]) &
        (df_filtered["date_parsed"].dt.date <= date_range[1])
    ]

# Key Performance Indicators
st.header("📈 Key Metrics")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        "Total NEOs",
        f"{len(df_filtered):,}",
        help="Number of NEOs matching your filters"
    )

with kpi2:
    high_risk_count = int((df_filtered["risk_label"] == "HIGH").sum()) if "risk_label" in df_filtered.columns else 0
    st.metric(
        "🔴 High Risk",
        high_risk_count,
        help="NEOs classified as HIGH risk"
    )

with kpi3:
    anomaly_count = int(df_filtered["is_anomaly"].sum()) if "is_anomaly" in df_filtered.columns else 0
    st.metric(
        "⚠️ Anomalies",
        anomaly_count,
        help="NEOs flagged as anomalous by Isolation Forest"
    )

with kpi4:
    avg_score = float(df_filtered["risk_score"].mean()) if len(df_filtered) > 0 else 0.0
    st.metric(
        "Avg Risk Score",
        f"{avg_score:.3f}",
        help="Average combined risk score"
    )

with kpi5:
    max_score_val = float(df_filtered["risk_score"].max()) if len(df_filtered) > 0 else 0.0
    st.metric(
        "Max Risk Score",
        f"{max_score_val:.3f}",
        help="Highest risk score in filtered data"
    )

# Visualizations
st.header("📊 Risk Analysis")

# Row 1: Risk distribution and model comparison
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.subheader("Risk Score Distribution")
    fig_hist = px.histogram(
        df_filtered,
        x="risk_score",
        nbins=40,
        color="risk_label" if "risk_label" in df_filtered.columns else None,
        color_discrete_map={"LOW": "#44ff44", "MEDIUM": "#ffaa00", "HIGH": "#ff4444"},
        labels={"risk_score": "Combined Risk Score", "count": "Number of NEOs"}
    )
    fig_hist.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Count",
        showlegend=True
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with viz_col2:
    st.subheader("Model Comparison: XGBoost vs Isolation Forest")
    if "xgb_risk_prob" in df_filtered.columns and "isolation_anomaly_score" in df_filtered.columns:
        fig_scatter_models = px.scatter(
            df_filtered,
            x="xgb_risk_prob",
            y="isolation_anomaly_score",
            color="risk_label" if "risk_label" in df_filtered.columns else None,
            color_discrete_map={"LOW": "#44ff44", "MEDIUM": "#ffaa00", "HIGH": "#ff4444"},
            hover_data=["name", "date", "risk_score"],
            labels={
                "xgb_risk_prob": "XGBoost Risk Probability",
                "isolation_anomaly_score": "Isolation Forest Anomaly Score"
            }
        )
        fig_scatter_models.update_layout(
            xaxis_title="XGBoost Risk Probability",
            yaxis_title="Isolation Forest Anomaly Score"
        )
        st.plotly_chart(fig_scatter_models, use_container_width=True)
    else:
        st.info("Model comparison requires both XGBoost and Isolation Forest scores")

# Row 2: Time series analysis
if df_filtered["date_parsed"].notna().any():
    st.subheader("📅 Risk Score Over Time")
    
    tmp = df_filtered.dropna(subset=["date_parsed"]).sort_values("date_parsed")
    
    # Aggregate by date (avg risk score per day)
    daily_risk = tmp.groupby(tmp["date_parsed"].dt.date).agg({
        "risk_score": ["mean", "max", "count"]
    }).reset_index()
    daily_risk.columns = ["date", "avg_risk", "max_risk", "neo_count"]
    
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=daily_risk["date"],
        y=daily_risk["avg_risk"],
        mode='lines+markers',
        name='Average Risk',
        line=dict(color='#4444ff', width=2),
        marker=dict(size=6)
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=daily_risk["date"],
        y=daily_risk["max_risk"],
        mode='lines',
        name='Maximum Risk',
        line=dict(color='#ff4444', width=2, dash='dash')
    ))
    
    fig_timeline.update_layout(
        xaxis_title="Date",
        yaxis_title="Risk Score",
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)

# Row 3: Physical characteristics vs risk
st.subheader("🌌 Physical Characteristics Analysis")

char_col1, char_col2 = st.columns(2)

with char_col1:
    if "miss_distance_km" in df_filtered.columns and "diameter_m" in df_filtered.columns:
        st.markdown("**Miss Distance vs Diameter (colored by risk)**")
        fig_scatter = px.scatter(
            df_filtered,
            x="miss_distance_km",
            y="diameter_m",
            color="risk_score",
            size="velocity_kmh" if "velocity_kmh" in df_filtered.columns else None,
            hover_data=["name", "date", "risk_label", "xgb_risk_prob", "isolation_anomaly_score"],
            labels={
                "miss_distance_km": "Miss Distance (km)",
                "diameter_m": "Diameter (m)",
                "risk_score": "Risk Score"
            },
            color_continuous_scale="RdYlGn_r"  # Red for high risk, green for low
        )
        fig_scatter.update_layout(
            xaxis_type="log",
            yaxis_type="log"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with char_col2:
    if "velocity_kmh" in df_filtered.columns:
        st.markdown("**Velocity Distribution by Risk Level**")
        fig_violin = px.violin(
            df_filtered,
            y="velocity_kmh",
            x="risk_label" if "risk_label" in df_filtered.columns else None,
            color="risk_label" if "risk_label" in df_filtered.columns else None,
            color_discrete_map={"LOW": "#44ff44", "MEDIUM": "#ffaa00", "HIGH": "#ff4444"},
            box=True,
            labels={"velocity_kmh": "Velocity (km/h)", "risk_label": "Risk Level"}
        )
        st.plotly_chart(fig_violin, use_container_width=True)

# High Risk Objects Table
st.header("⚠️ Highest Risk Objects")

# Select display columns
display_cols = [
    c for c in [
        "date", "name", "neo_reference_id", "diameter_m", 
        "miss_distance_km", "velocity_kmh", "xgb_risk_prob",
        "isolation_anomaly_score", "is_anomaly", "risk_score", 
        "risk_label", "hazardous"
    ] if c in df_filtered.columns
]

# Sort by risk score
top_risks = df_filtered.sort_values("risk_score", ascending=False).head(50)

# Apply color coding to risk labels
def color_risk_label(val):
    if val == "HIGH":
        return 'background-color: #ff444440'
    elif val == "MEDIUM":
        return 'background-color: #ffaa0040'
    elif val == "LOW":
        return 'background-color: #44ff4440'
    return ''

if "risk_label" in top_risks.columns:
    styled_df = top_risks[display_cols].style.applymap(
        color_risk_label, 
        subset=["risk_label"]
    ).format({
        "risk_score": "{:.3f}",
        "xgb_risk_prob": "{:.3f}",
        "isolation_anomaly_score": "{:.3f}",
        "diameter_m": "{:.1f}",
        "miss_distance_km": "{:,.0f}",
        "velocity_kmh": "{:,.0f}"
    })
    st.dataframe(styled_df, use_container_width=True, height=400)
else:
    st.dataframe(top_risks[display_cols], use_container_width=True, height=400)

# Model Information
with st.expander("ℹ️ About the Models"):
    st.markdown("""
    ### Risk Assessment Methodology
    
    This dashboard combines two complementary machine learning approaches:
    
    **1. XGBoost Classifier**
    - Predicts the probability that an NEO is potentially hazardous
    - Trained on labeled NASA data with known hazard classifications
    - Considers diameter, velocity, miss distance, and orbital parameters
    - Target accuracy: ≥85%
    
    **2. Isolation Forest (Anomaly Detection)**
    - Identifies unusual or extreme NEOs that don't fit normal patterns
    - Unsupervised learning - doesn't require labeled data
    - Flags objects with unexpected combinations of characteristics
    - Helps catch edge cases the classifier might miss
    
    **Combined Risk Score**
    - Weighted combination: 60% XGBoost + 40% Isolation Forest
    - Risk Labels:
        - 🟢 **LOW**: Risk score < 0.3
        - 🟡 **MEDIUM**: Risk score 0.3 - 0.7
        - 🔴 **HIGH**: Risk score > 0.7
    
    **Data Source**: NASA Near-Earth Object Web Service (NeoWs) API
    """)

# Key Terms Glossary
with st.expander("📖 Glossary"):
    st.markdown("""
    **Near-Earth Object (NEO)**: An asteroid or comet with an orbit that brings it within 1.3 AU of the Sun and within 0.3 AU (45 million km) of Earth
    
    **Potentially Hazardous Asteroid (PHA)**: An NEO with a Minimum Orbit Intersection Distance (MOID) < 0.05 AU and absolute magnitude H ≤ 22.0 (diameter ≥ ~140m)
    
    **Miss Distance**: The closest distance the NEO will come to Earth during its approach
    
    **Absolute Magnitude (H)**: A measure of the asteroid's intrinsic brightness; lower values indicate larger objects
    
    **Risk Score**: Combined probability (0-1) that an object poses a threat, calculated from both ML models
    
    **Anomaly Score**: Measure of how unusual an NEO's characteristics are compared to the broader population
    """)

# Footer
st.divider()
st.caption(f"""
**Dashboard Auto-refreshes every {AUTO_REFRESH_SECONDS} seconds** | 
Data updates when NASA's NeoWs API provides new information |
For questions or feedback, refer to your project documentation
""")

# Auto-refresh indicator
placeholder = st.empty()
with placeholder:
    st.info(f"⏱️ Dashboard will auto-refresh in {AUTO_REFRESH_SECONDS} seconds...")
