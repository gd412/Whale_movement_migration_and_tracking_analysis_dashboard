"""
AIS Collision Risk Assessment Page - AISStream.io WebSocket Version
Streamlit dashboard for monitoring ship-whale collision risks
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
import logging

from ais_utils import AISStreamFetcher, load_whale_data, create_whale_zones
from risk_calculator import (
    assess_collision_risks,
    get_risk_statistics,
    get_risk_color
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ais_collision_page():
    """Main function for AIS Collision Risk page"""
    
    st.title("🚢 Real-Time Ship-Whale Collision Risk Monitor")
    
    st.markdown("""
    This system monitors live vessel traffic using **AISStream.io** (free WebSocket API) 
    and compares it with whale habitat zones to identify potential collision risks.
    """)
    
    # ================================================================
    # LOAD CONFIGURATION
    # ================================================================
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        st.error("❌ config.json not found. Please create it with your AISStream API key.")
        st.stop()
    
    # Check if API is configured
    if config['ais_api']['api_key'] == 'YOUR_AISSTREAM_API_KEY':
        st.error("❌ Please configure your AISStream.io API key in config.json")
        st.info("""
        **How to get your FREE API key:**
        1. Visit: https://aisstream.io/
        2. Click "Sign in with GitHub"
        3. Authorize the app
        4. Copy your API key from the dashboard
        5. Paste it into config.json
        """)
        st.stop()
    
    # ================================================================
    # INITIALIZE SESSION STATE
    # ================================================================
    
    if 'vessels_df' not in st.session_state:
        st.session_state.vessels_df = pd.DataFrame()
    
    if 'whale_zones_df' not in st.session_state:
        st.session_state.whale_zones_df = pd.DataFrame()
    
    if 'risk_df' not in st.session_state:
        st.session_state.risk_df = pd.DataFrame()
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    
    # ================================================================
    # SIDEBAR CONTROLS
    # ================================================================
    
    st.sidebar.header("🎛️ Controls")
    
    # Geographic area selection
    st.sidebar.subheader("Geographic Area")
    
    bounds = config['geographic_bounds']
    
    lat_min = st.sidebar.number_input(
        "Latitude Min",
        value=bounds['lat_min'],
        min_value=-90.0,
        max_value=90.0,
        step=0.5
    )
    
    lat_max = st.sidebar.number_input(
        "Latitude Max",
        value=bounds['lat_max'],
        min_value=-90.0,
        max_value=90.0,
        step=0.5
    )
    
    lon_min = st.sidebar.number_input(
        "Longitude Min",
        value=bounds['lon_min'],
        min_value=-180.0,
        max_value=180.0,
        step=0.5
    )
    
    lon_max = st.sidebar.number_input(
        "Longitude Max",
        value=bounds['lon_max'],
        min_value=-180.0,
        max_value=180.0,
        step=0.5
    )
    
    # Collection duration
    st.sidebar.subheader("Data Collection")
    
    collection_duration = st.sidebar.slider(
        "Collection Duration (seconds)",
        min_value=10,
        max_value=120,
        value=config['websocket_settings'].get('data_collection_duration_seconds', 60),
        step=10,
        help="How long to collect AIS data via WebSocket"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Fetch Live Data", type="primary"):
        fetch_data(config, lat_min, lat_max, lon_min, lon_max, collection_duration)
    
    # Info about API
    st.sidebar.info("""
    ℹ️ **AISStream.io WebSocket**
    - Completely FREE
    - Real-time streaming
    - Global coverage
    - No rate limits
    """)
    
    # Display last update time
    if st.session_state.last_update:
        st.sidebar.success(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    else:
        st.sidebar.warning("⚠️ No data loaded yet - click 'Fetch Live Data'")
    
    # ================================================================
    # METRICS DASHBOARD
    # ================================================================
    
    st.header("📊 Current Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Vessels",
            len(st.session_state.vessels_df)
        )
    
    with col2:
        st.metric(
            "Whale Zones",
            len(st.session_state.whale_zones_df)
        )
    
    with col3:
        risk_count = len(st.session_state.risk_df)
        st.metric(
            "At-Risk Vessels",
            risk_count,
            delta=None if risk_count == 0 else "⚠️"
        )
    
    with col4:
        if not st.session_state.risk_df.empty:
            critical_high = len(
                st.session_state.risk_df[
                    st.session_state.risk_df['risk_level'].isin(['CRITICAL', 'HIGH'])
                ]
            )
            st.metric(
                "High Priority",
                critical_high,
                delta="Alert" if critical_high > 0 else None,
                delta_color="inverse"
            )
        else:
            st.metric("High Priority", 0)
    
    # ================================================================
    # INTERACTIVE MAP
    # ================================================================
    
    st.header("🗺️ Live Collision Risk Map")
    
    if st.session_state.last_update:
        fig = create_map(
            st.session_state.vessels_df,
            st.session_state.whale_zones_df,
            st.session_state.risk_df,
            lat_min, lat_max, lon_min, lon_max
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Click 'Fetch Live Data' in the sidebar to load vessel positions")
        
        # Show placeholder map
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lat=[(lat_min + lat_max) / 2],
            lon=[(lon_min + lon_max) / 2],
            mode='markers',
            marker=dict(size=1, color='gray'),
            showlegend=False
        ))
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=(lat_min + lat_max) / 2, lon=(lon_min + lon_max) / 2),
                zoom=4
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ================================================================
    # RISK ALERTS TABLE
    # ================================================================
    
    st.header("⚠️ Risk Alerts")
    
    if not st.session_state.risk_df.empty:
        
        # Sort by risk score
        display_df = st.session_state.risk_df.sort_values('risk_score', ascending=False)
        
        # Add color indicators
        def color_risk_level(val):
            color = get_risk_color(val)
            return f'background-color: {color}'
        
        # Display table with styling
        st.dataframe(
            display_df[[
                'vessel_name', 'mmsi', 'risk_level', 'risk_score',
                'distance_km', 'vessel_speed', 'timestamp'
            ]].style.applymap(
                color_risk_level,
                subset=['risk_level']
            ),
            use_container_width=True
        )
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Risk Report (CSV)",
            csv,
            f"collision_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
        
        # Risk statistics
        st.subheader("📈 Risk Statistics")
        
        stats = get_risk_statistics(st.session_state.risk_df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Critical", stats['critical'])
        with col2:
            st.metric("High", stats['high'])
        with col3:
            st.metric("Medium", stats['medium'])
        with col4:
            st.metric("Low", stats['low'])
        with col5:
            st.metric("Avg Distance", f"{stats['avg_distance']} km")
    
    else:
        if st.session_state.last_update:
            st.success("✅ No collision risks detected")
        else:
            st.info("No data loaded yet. Click 'Fetch Live Data' to begin monitoring.")
    
    # ================================================================
    # DETAILED VESSEL INFO
    # ================================================================
    
    if not st.session_state.vessels_df.empty:
        with st.expander("🚢 All Vessels in Area"):
            st.dataframe(
                st.session_state.vessels_df[[
                    'mmsi', 'vessel_name', 'latitude', 'longitude',
                    'speed_knots', 'course', 'destination'
                ]],
                use_container_width=True
            )


def fetch_data(config, lat_min, lat_max, lon_min, lon_max, duration):
    """
    Fetch AIS data via WebSocket and calculate risks
    
    Args:
        config: Configuration dictionary
        lat_min, lat_max, lon_min, lon_max: Geographic bounds
        duration: Collection duration in seconds
    """
    
    # Create progress container
    progress_container = st.empty()
    
    with progress_container.container():
        st.info(f"🌊 Connecting to AISStream.io WebSocket... (collecting for {duration}s)")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize AIS fetcher
        ais_fetcher = AISStreamFetcher(config)
        
        # Start data collection
        import time
        start_time = time.time()
        
        # Update progress while collecting
        # (In practice, WebSocket runs async, so this is approximate)
        for i in range(duration):
            if time.time() - start_time >= duration:
                break
            progress_bar.progress((i + 1) / duration)
            status_text.text(f"Collecting AIS data... {i+1}/{duration} seconds")
            time.sleep(1)
        
        # Fetch vessels
        vessels_df = ais_fetcher.fetch_vessels(
            lat_min, lat_max, lon_min, lon_max, duration
        )
        
        progress_bar.progress(100)
    
    # Clear progress
    progress_container.empty()
    
    if vessels_df.empty:
        st.warning("""
        ⚠️ No vessels found in this area. 
        
        **Possible reasons:**
        - No ship traffic in this geographic area
        - Try a coastal area or major shipping lane
        - Increase collection duration
        - Check API key is valid
        """)
        return
    
    # Filter to high-risk vessels
    vessels_df = ais_fetcher.filter_high_risk_vessels(vessels_df)
    
    st.session_state.vessels_df = vessels_df
    st.success(f"✅ Found {len(vessels_df)} vessels")
    
    with st.spinner("🐋 Loading whale habitat data..."):
        
        # Load whale data
        whale_df = load_whale_data(config['whale_data_path'])
        
        if whale_df.empty:
            st.error("Failed to load whale data")
            return
        
        # Create whale zones
        whale_zones_df = create_whale_zones(whale_df, grid_size=0.5)
        
        st.session_state.whale_zones_df = whale_zones_df
        st.success(f"✅ Identified {len(whale_zones_df)} whale concentration zones")
    
    with st.spinner("⚠️ Calculating collision risks..."):
        
        # Assess risks
        risk_df = assess_collision_risks(vessels_df, whale_zones_df, config)
        
        st.session_state.risk_df = risk_df
        
        if not risk_df.empty:
            high_risk_count = len(risk_df[risk_df['risk_level'].isin(['CRITICAL', 'HIGH'])])
            if high_risk_count > 0:
                st.warning(f"⚠️ {high_risk_count} high-priority collision risks detected!")
            else:
                st.info(f"Found {len(risk_df)} vessels within monitoring range")
        else:
            st.success("✅ No collision risks detected")
    
    # Update timestamp
    st.session_state.last_update = datetime.now()


def create_map(vessels_df, whale_zones_df, risk_df, lat_min, lat_max, lon_min, lon_max):
    """
    Create interactive Plotly map
    
    Args:
        vessels_df: Vessel positions
        whale_zones_df: Whale zones
        risk_df: Risk assessments
        lat_min, lat_max, lon_min, lon_max: Map bounds
        
    Returns:
        Plotly Figure object
    """
    
    fig = go.Figure()
    
    # Plot whale zones (blue circles)
    if not whale_zones_df.empty:
        fig.add_trace(go.Scattermapbox(
            lat=whale_zones_df['latitude'],
            lon=whale_zones_df['longitude'],
            mode='markers',
            marker=dict(
                size=12,
                color='blue',
                opacity=0.6
            ),
            name='Whale Zones',
            text=[f"Density: {d}" for d in whale_zones_df['density']],
            hovertemplate='<b>Whale Zone</b><br>%{text}<br>%{lat:.2f}, %{lon:.2f}<extra></extra>'
        ))
    
    # Plot vessels with risk coloring
    if not vessels_df.empty:
        
        # Create risk level mapping
        if not risk_df.empty:
            risk_lookup = dict(zip(risk_df['mmsi'], risk_df['risk_level']))
            vessels_df['risk_level'] = vessels_df['mmsi'].map(risk_lookup).fillna('SAFE')
            vessels_df['risk_color'] = vessels_df['risk_level'].apply(get_risk_color)
        else:
            vessels_df['risk_level'] = 'SAFE'
            vessels_df['risk_color'] = 'green'
        
        # Plot each risk level separately for better legend
        for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'SAFE']:
            level_vessels = vessels_df[vessels_df['risk_level'] == risk_level]
            
            if not level_vessels.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=level_vessels['latitude'],
                    lon=level_vessels['longitude'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=get_risk_color(risk_level)
                    ),
                    name=f'{risk_level} Risk',
                    text=[
                        f"{row['vessel_name']}<br>MMSI: {row['mmsi']}<br>Speed: {row.get('speed_knots', 0):.1f} kts"
                        for _, row in level_vessels.iterrows()
                    ],
                    hovertemplate='<b>%{text}</b><br>%{lat:.3f}, %{lon:.3f}<extra></extra>'
                ))
    
    # Map layout
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=(lat_min + lat_max) / 2,
                lon=(lon_min + lon_max) / 2
            ),
            zoom=5
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )
    
    return fig

def run_ais_collision():
    return run_ais_collision_page()
