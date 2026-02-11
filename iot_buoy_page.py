import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random

# =========================================================
# SMART IOT BUOY SIMULATION PAGE
# =========================================================

def run_iot_buoy():

    st.title("📡 Smart IoT Buoy — Live Whale Detection Simulator")

    st.info("Simulated real-time ocean sensor buoys sending whale detection signals")

    # -----------------------------------------------------
    # SESSION STATE INIT
    # -----------------------------------------------------
    if "buoy_data" not in st.session_state:
        st.session_state.buoy_data = []

    if "running" not in st.session_state:
        st.session_state.running = False

    # -----------------------------------------------------
    # SIDEBAR CONTROLS
    # -----------------------------------------------------
    st.sidebar.subheader("⚙ IoT Controls")

    update_speed = st.sidebar.slider(
        "Update interval (seconds)",
        1, 5, 2
    )

    num_buoys = st.sidebar.slider(
        "Number of Buoys",
        1, 5, 3
    )

    colA, colB = st.columns(2)

    if colA.button("▶ Start Simulation"):
        st.session_state.running = True

    if colB.button("⏹ Stop Simulation"):
        st.session_state.running = False

    # -----------------------------------------------------
    # SENSOR SIMULATION FUNCTION
    # -----------------------------------------------------
    def generate_buoy_reading(buoy_id):
        base_lat = 34.0
        base_lon = -120.0

        lat = base_lat + random.uniform(-2, 2)
        lon = base_lon + random.uniform(-2, 2)

        temp = random.uniform(5, 18)
        depth = random.uniform(50, 300)
        sound = random.uniform(20, 120)
        wave = random.uniform(0.5, 3.5)

        # simple whale likelihood logic
        whale_prob = 0

        if sound > 85:
            whale_prob += 0.4
        if 8 < temp < 16:
            whale_prob += 0.3
        if depth > 80:
            whale_prob += 0.2
        if wave < 2:
            whale_prob += 0.1

        whale_prob = min(1.0, whale_prob + random.uniform(0, 0.2))

        return {
            "time": pd.Timestamp.now(),
            "buoy": f"B-{buoy_id}",
            "lat": lat,
            "lon": lon,
            "temp": round(temp, 2),
            "depth": round(depth, 1),
            "sound": round(sound, 1),
            "wave": round(wave, 2),
            "whale_prob": round(whale_prob, 2)
        }

    # -----------------------------------------------------
    # LIVE LOOP
    # -----------------------------------------------------
    placeholder_metrics = st.empty()
    placeholder_map = st.empty()
    placeholder_chart = st.empty()
    placeholder_table = st.empty()

    if st.session_state.running:

        for _ in range(50):  # loop batch updates

            new_rows = []
            for i in range(num_buoys):
                new_rows.append(generate_buoy_reading(i+1))

            st.session_state.buoy_data.extend(new_rows)

            df = pd.DataFrame(st.session_state.buoy_data)

            # -------------------------
            # METRICS
            # -------------------------
            with placeholder_metrics.container():
                st.subheader("📊 Live Sensor Metrics")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Signals", len(df))
                c2.metric("Active Buoys", df["buoy"].nunique())
                c3.metric(
                    "Whale Alerts",
                    (df["whale_prob"] > 0.7).sum()
                )

            # -------------------------
            # LIVE MAP
            # -------------------------
            with placeholder_map.container():
                st.subheader("🗺 Live Buoy Positions")

                latest = df.sort_values("time").groupby("buoy").tail(1)

                fig = px.scatter_mapbox(
                    latest,
                    lat="lat",
                    lon="lon",
                    color="whale_prob",
                    size="whale_prob",
                    zoom=3,
                    height=500
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)

            # -------------------------
            # SOUND LEVEL CHART
            # -------------------------
            with placeholder_chart.container():
                st.subheader("🔊 Sound Level Trend")

                fig2 = px.line(
                    df.tail(100),
                    x="time",
                    y="sound",
                    color="buoy"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # -------------------------
            # TABLE
            # -------------------------
            with placeholder_table.container():
                st.subheader("📋 Latest Readings")
                st.dataframe(df.tail(10))

            time.sleep(update_speed)

            if not st.session_state.running:
                break

    else:
        st.warning("Simulation stopped — press Start")

