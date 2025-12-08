# ================================================================
# 🐋 Whale Movement Tracking & Migration Analysis Dashboard (Final)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# ---------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Whale Movement Tracking System",
    page_icon="🐋",
    layout="wide"
)

st.title("🐋 Whale Movement Tracking & Migration Analysis Dashboard")
st.markdown("### Dataset: Blue Whales – Eastern North Pacific (1993–2008)")

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    file_path = r"C:\Users\laptech\Desktop\Whale Movement Analysis\Blue whales Eastern North Pacific 1993-2008 - Argos Data.csv"
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # Detect columns
    lat_col = [c for c in df.columns if 'lat' in c.lower()]
    lon_col = [c for c in df.columns if 'lon' in c.lower()]
    time_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

    if lat_col: df.rename(columns={lat_col[0]: "Latitude"}, inplace=True)
    if lon_col: df.rename(columns={lon_col[0]: "Longitude"}, inplace=True)
    if time_col: df.rename(columns={time_col[0]: "Timestamp"}, inplace=True)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[(df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180))]
    df = df.sort_values("Timestamp")
    return df

df = load_data()
st.success(f"✅ Data loaded successfully — {len(df)} records")

# ---------------------------------------------------------------
# Overview Metrics
# ---------------------------------------------------------------
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
if "Timestamp" in df.columns:
    col2.metric("Date Range", f"{df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")
col3.metric("Latitude Range", f"{df['Latitude'].min():.2f} → {df['Latitude'].max():.2f}")
col4.metric("Longitude Range", f"{df['Longitude'].min():.2f} → {df['Longitude'].max():.2f}")

st.dataframe(df.head())

# ---------------------------------------------------------------
# Animated Migration Map
# ---------------------------------------------------------------
st.header("🌊 Animated Whale Migration Path")

fig_ani = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color_discrete_sequence=["deepskyblue"],
    hover_data=["Timestamp"],
    animation_frame=df["Timestamp"].dt.strftime("%Y-%m-%d") if "Timestamp" in df.columns else None,
    zoom=3,
    height=650
)
fig_ani.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_ani, use_container_width=True)

# ---------------------------------------------------------------
# Distance & Speed Estimation
# ---------------------------------------------------------------
st.header("📏 Distance & Speed Analysis")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(min(1, a)))

df["Distance_km"] = np.nan
for i in range(1, len(df)):
    lat1, lon1 = df.iloc[i - 1][["Latitude", "Longitude"]]
    lat2, lon2 = df.iloc[i][["Latitude", "Longitude"]]
    df.loc[df.index[i], "Distance_km"] = haversine(lat1, lon1, lat2, lon2)

# Speed = distance/time
df["Time_diff_hr"] = df["Timestamp"].diff().dt.total_seconds() / 3600
df["Speed_kmph"] = df["Distance_km"] / df["Time_diff_hr"]
df["Speed_kmph"] = df["Speed_kmph"].clip(upper=50)

col1, col2 = st.columns(2)
col1.metric("Total Distance Covered (km)", f"{df['Distance_km'].sum():.2f}")
col2.metric("Average Speed (km/h)", f"{df['Speed_kmph'].mean():.2f}")

fig_speed = px.line(df, x="Timestamp", y="Speed_kmph", title="🐳 Whale Speed Over Time", markers=True)
st.plotly_chart(fig_speed, use_container_width=True)

# ---------------------------------------------------------------
# Monthly Activity Trend
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
# Hotspot Detection
# ---------------------------------------------------------------
st.header("🔥 Migration Hotspots")

num_clusters = st.slider("Select number of clusters (k):", 3, 10, 5)
coords = df[["Latitude", "Longitude"]].dropna()
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(coords)

fig_cluster = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="Cluster",
    zoom=3,
    height=600,
    mapbox_style="carto-positron",
    title="Whale Hotspot Clusters"
)
st.plotly_chart(fig_cluster, use_container_width=True)

cluster_counts = df["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Points"]
st.dataframe(cluster_counts)

# ---------------------------------------------------------------
# 🧠 Future Predictions with Prophet (Enhanced)
# ---------------------------------------------------------------
st.header("📈 Advanced Future Predictions (Prophet Model)")

if "Timestamp" in df.columns:
    df_forecast = df.groupby(df["Timestamp"].dt.date)["Distance_km"].sum().reset_index()
    df_forecast.columns = ["ds", "y"]

    # Train Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    st.subheader("🔮 30-Day Future Migration Forecast")
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("📊 Trend & Seasonality Insights")
    fig_components = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_components, use_container_width=True)

    st.write("**📅 Forecast Data (Preview):**")
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(15))

    # Summary cards
    avg_pred = forecast["yhat"].tail(15).mean()
    max_pred = forecast["yhat"].tail(15).max()
    min_pred = forecast["yhat"].tail(15).min()

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Avg Daily Movement (km)", f"{avg_pred:.2f}")
    col2.metric("Max Expected Movement (km)", f"{max_pred:.2f}")
    col3.metric("Min Expected Movement (km)", f"{min_pred:.2f}")

    st.info(
        "The Prophet model forecasts future whale migration distances based on past travel patterns. "
        "These insights can help marine researchers identify likely migration periods and predict seasonal behavior."
    )

else:
    st.warning("⏰ Timestamp column not available for forecasting.")

# ---------------------------------------------------------------
# Footer
# ---------------------------------------------------------------
st.markdown("---")
st.caption("Developed by **Banu Varshit B** | Blue Whale Movement Analysis | 2025")
