# ================================================================
# 🛡️ Conservation Decision Support Dashboard
# Advanced analytics derived from whale movement dataset
# Plug-in page — no dependency changes to other modules
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt


DATA_PATH = r"C:\Users\laptech\OneDrive\Desktop\Whale Movement Analysis\Blue whales Eastern North Pacific 1993-2008 - Argos Data.csv"


# ================================================================
# DATA LOADER
# ================================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    lat_col = [c for c in df.columns if 'lat' in c.lower()][0]
    lon_col = [c for c in df.columns if 'lon' in c.lower()][0]
    time_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]

    df.rename(columns={
        lat_col: "Latitude",
        lon_col: "Longitude",
        time_col: "Timestamp"
    }, inplace=True)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df.sort_values("Timestamp")

    return df


# ================================================================
# DISTANCE FUNCTION
# ================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(min(1, a)))


# ================================================================
# MAIN PAGE FUNCTION
# ================================================================

def run_conservation_dashboard():

    st.title("🛡️ Conservation Decision Support Dashboard")

    st.markdown("""
    Advanced conservation analytics derived directly from whale tracking data.
    Helps identify priority protection regions, migration pressure zones,
    and seasonal conservation timing windows.
    """)

    df = load_data()

    # ============================================================
    # METRICS
    # ============================================================

    st.header("📊 Conservation Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Tracking Records", len(df))
    c2.metric("Years Covered", df["Timestamp"].dt.year.nunique())
    c3.metric("Unique Months Active", df["Timestamp"].dt.month.nunique())

    # ============================================================
    # FEATURE 1 — PROTECTED ZONE CLUSTER GENERATOR
    # ============================================================

    st.header("📍 Recommended Protected Zones (Cluster Based)")

    k = st.slider("Number of Protection Zones", 3, 12, 6)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["Zone"] = kmeans.fit_predict(df[["Latitude", "Longitude"]])

    zone_summary = df.groupby("Zone").agg({
        "Latitude": "mean",
        "Longitude": "mean",
        "Timestamp": "count"
    }).rename(columns={"Timestamp": "Activity Points"}).reset_index()

    fig = px.scatter_mapbox(
        zone_summary,
        lat="Latitude",
        lon="Longitude",
        size="Activity Points",
        color="Activity Points",
        zoom=3,
        height=600,
        title="Protection Zone Candidates"
    )

    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(zone_summary)

    # ============================================================
    # FEATURE 2 — MIGRATION CORRIDOR DETECTION
    # ============================================================

    st.header("🛣️ Major Migration Corridors")

    df_sorted = df.sort_values("Timestamp").copy()
    df_sorted["Distance_km"] = np.nan

    for i in range(1, len(df_sorted)):
        lat1, lon1 = df_sorted.iloc[i-1][["Latitude", "Longitude"]]
        lat2, lon2 = df_sorted.iloc[i][["Latitude", "Longitude"]]
        df_sorted.loc[df_sorted.index[i], "Distance_km"] = haversine(lat1, lon1, lat2, lon2)

    top_moves = df_sorted.nlargest(200, "Distance_km")

    fig2 = px.scatter_mapbox(
        top_moves,
        lat="Latitude",
        lon="Longitude",
        color="Distance_km",
        size="Distance_km",
        zoom=3,
        height=600,
        title="High Movement Corridor Points"
    )

    fig2.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # FEATURE 3 — SEASONAL CONSERVATION WINDOWS
    # ============================================================

    st.header("📆 Seasonal Protection Windows")

    df["Month"] = df["Timestamp"].dt.month
    seasonal = df["Month"].value_counts().sort_index().reset_index()
    seasonal.columns = ["Month", "Activity"]

    fig3 = px.bar(
        seasonal,
        x="Month",
        y="Activity",
        title="Whale Activity by Month"
    )

    st.plotly_chart(fig3, use_container_width=True)

    peak_months = seasonal.sort_values("Activity", ascending=False).head(3)

    st.success("Top 3 High-Priority Conservation Months:")
    st.dataframe(peak_months)

    # ============================================================
    # FEATURE 4 — ZONE STABILITY SCORE
    # ============================================================

    st.header("🧭 Zone Stability Score")

    stability = df.groupby("Zone").agg({
        "Latitude": "std",
        "Longitude": "std"
    }).reset_index()

    stability["StabilityScore"] = 1 / (stability["Latitude"] + stability["Longitude"])

    st.dataframe(stability)

    fig4 = px.bar(
        stability,
        x="Zone",
        y="StabilityScore",
        title="Zone Stability (Higher = More Consistent Whale Presence)"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # ============================================================
    # FEATURE 5 — EXPORT CONSERVATION ZONES
    # ============================================================

    st.header("📥 Export Protection Zones")

    csv = zone_summary.to_csv(index=False)

    st.download_button(
        "Download Protection Zones CSV",
        csv,
        "protected_zones.csv",
        "text/csv"
    )

    
    
