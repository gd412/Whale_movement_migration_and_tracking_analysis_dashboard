# ================================================================
# 🐋 Whale Movement Tracking + AI Vision
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Whale Movement Tracking System",
    page_icon="🐋",
    layout="wide"
)

# ================================================================
# SIDEBAR — ONLY INTEGRATION (3RD PAGE ADDED)
# ================================================================
page = st.sidebar.selectbox(
    "Select Module",
    [
        "Movement Analytics",
        "Whale Image AI Classifier",
        "Smart IoT Simulation"
    ]
)

# ================================================================
DATA_PATH = r"C:\Users\laptech\OneDrive\Desktop\Whale Movement Analysis\Blue whales Eastern North Pacific 1993-2008 - Argos Data.csv"
# ================================================================


# ================================================================
# 🐋 PAGE 1 — MOVEMENT ANALYTICS
# ================================================================
if page == "Movement Analytics":

    st.title("🐋 Whale Movement Tracking & Migration Analysis Dashboard")

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

    df = load_data()

    # ---------------------------------------------------------------
    # Dataset Overview
    # ---------------------------------------------------------------
    st.header("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Date Range",
              f"{df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")
    c3.metric("Latitude Range",
              f"{df['Latitude'].min():.2f} → {df['Latitude'].max():.2f}")
    c4.metric("Longitude Range",
              f"{df['Longitude'].min():.2f} → {df['Longitude'].max():.2f}")

    st.dataframe(df.head())

    # ---------------------------------------------------------------
    # Animated Map
    # ---------------------------------------------------------------
    st.header("🌊 Animated Whale Migration Path")

    fig_ani = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        hover_data=["Timestamp"],
        animation_frame=df["Timestamp"].dt.strftime("%Y-%m-%d"),
        zoom=3,
        height=650
    )
    fig_ani.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_ani, use_container_width=True)

    # ---------------------------------------------------------------
    # Distance & Speed
    # ---------------------------------------------------------------
    st.header("📏 Distance & Speed Analysis")

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 2 * R * asin(sqrt(min(1, a)))

    df["Distance_km"] = np.nan

    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i-1][["Latitude", "Longitude"]]
        lat2, lon2 = df.iloc[i][["Latitude", "Longitude"]]
        df.loc[df.index[i], "Distance_km"] = haversine(lat1, lon1, lat2, lon2)

    df["Time_diff_hr"] = df["Timestamp"].diff().dt.total_seconds() / 3600
    df["Speed_kmph"] = df["Distance_km"] / df["Time_diff_hr"]
    df["Speed_kmph"] = df["Speed_kmph"].clip(upper=50)

    st.plotly_chart(
        px.line(df, x="Timestamp", y="Speed_kmph", markers=True),
        use_container_width=True
    )

    # ---------------------------------------------------------------
    # ✅ MONTHLY ACTIVITY TREND — EXACT BLOCK YOU GAVE
    # ---------------------------------------------------------------
    st.header("📆 Monthly Movement Trend")

    if "Timestamp" in df.columns:
        df["Month"] = df["Timestamp"].dt.to_period("M")
        monthly_dist = df.groupby("Month")["Distance_km"].sum().reset_index()
        monthly_dist["Month"] = monthly_dist["Month"].astype(str)

        fig_month = px.bar(
            monthly_dist,
            x="Month",
            y="Distance_km",
            title="Monthly Migration Distance",
            color="Distance_km",
            color_continuous_scale="teal"
        )
        st.plotly_chart(fig_month, use_container_width=True)

    # ---------------------------------------------------------------
    # Hotspots
    # ---------------------------------------------------------------
    st.header("🔥 Migration Hotspots")

    k = st.slider("Clusters", 3, 10, 5)
    coords = df[["Latitude", "Longitude"]]

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    df["Cluster"] = kmeans.fit_predict(coords)

    fig_cluster = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        zoom=3,
        height=600
    )
    fig_cluster.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_cluster, use_container_width=True)

    # ---------------------------------------------------------------
    # Forecast
    # ---------------------------------------------------------------
    st.header("📈 Future Movement Forecast")

    df_fore = df.groupby(df["Timestamp"].dt.date)["Distance_km"].sum().reset_index()
    df_fore.columns = ["ds", "y"]

    model = Prophet(yearly_seasonality=True)
    model.fit(df_fore)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)


# ================================================================
# PAGE 2 — CLASSIFIER (UNCHANGED)
# ================================================================
elif page == "Whale Image AI Classifier":
    from image_ai_page import run_image_ai
    run_image_ai()


# ================================================================
# PAGE 3 — IOT PAGE (YOUR CODE — IMPORT ONLY)
# ================================================================
elif page == "Smart IoT Simulation":
    from iot_buoy_page import run_iot_buoy
    run_iot_buoy()


st.markdown("---")
st.caption("Whale tracking AI System")
