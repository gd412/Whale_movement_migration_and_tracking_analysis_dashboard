"""
AIS Data Fetching Utilities - AISStream.io WebSocket
Fetches real-time vessel data from AISStream.io (Free WebSocket API)
"""

import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AISStreamFetcher:
    """Fetch and process AIS vessel data from AISStream.io WebSocket"""
    
    def __init__(self, config):
        """
        Initialize AIS stream fetcher
        
        Args:
            config: Dictionary with AIS API configuration
        """
        self.config = config
        self.api_key = config['ais_api']['api_key']
        self.websocket_url = config['ais_api']['websocket_url']
        self.vessels = []
        self.is_collecting = False
        
    def fetch_vessels(self, lat_min, lat_max, lon_min, lon_max, duration_seconds=60):
        """
        Fetch vessels in specified geographic area using WebSocket
        
        Args:
            lat_min, lat_max: Latitude bounds
            lon_min, lon_max: Longitude bounds
            duration_seconds: How long to collect data (default 60 seconds)
            
        Returns:
            DataFrame with vessel data
        """
        
        logger.info(f"Starting AIS data collection for {duration_seconds} seconds")
        logger.info(f"Area: {lat_min},{lon_min} to {lat_max},{lon_max}")
        
        # Reset vessels list
        self.vessels = []
        
        # Run WebSocket connection in asyncio event loop
        try:
            asyncio.run(self._collect_ais_data(
                lat_min, lat_max, lon_min, lon_max, duration_seconds
            ))
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            return pd.DataFrame()
        
        # Convert collected vessels to DataFrame
        if not self.vessels:
            logger.warning("No vessels collected")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.vessels)
        
        # Remove duplicates (same vessel reported multiple times)
        df = df.drop_duplicates(subset=['mmsi'], keep='last')
        
        logger.info(f"Collected {len(df)} unique vessels")
        
        return df
    
    async def _collect_ais_data(self, lat_min, lat_max, lon_min, lon_max, duration):
        """
        Async function to collect AIS data via WebSocket
        
        Args:
            lat_min, lat_max, lon_min, lon_max: Geographic bounds
            duration: Collection duration in seconds
        """
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                
                # Create subscription message
                subscription = {
                    "APIKey": self.api_key,
                    "BoundingBoxes": [
                        [[lat_min, lon_min], [lat_max, lon_max]]
                    ],
                    "FilterMessageTypes": ["PositionReport"]  # Only position updates
                }
                
                # Send subscription within 3 seconds (required by API)
                await websocket.send(json.dumps(subscription))
                logger.info("Subscription sent, collecting data...")
                
                self.is_collecting = True
                start_time = asyncio.get_event_loop().time()
                
                # Collect messages for specified duration
                while self.is_collecting:
                    try:
                        # Check if duration exceeded
                        if asyncio.get_event_loop().time() - start_time > duration:
                            logger.info("Collection duration reached")
                            break
                        
                        # Wait for message with timeout
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=5.0
                        )
                        
                        # Parse message
                        data = json.loads(message)
                        
                        # Process vessel data
                        vessel = self._parse_ais_message(data)
                        if vessel:
                            self.vessels.append(vessel)
                            
                            # Log progress every 10 vessels
                            if len(self.vessels) % 10 == 0:
                                logger.info(f"Collected {len(self.vessels)} vessels...")
                        
                        # Stop if we've collected enough vessels
                        max_vessels = self.config['websocket_settings'].get(
                            'max_vessels_to_collect', 500
                        )
                        if len(self.vessels) >= max_vessels:
                            logger.info(f"Reached max vessels limit ({max_vessels})")
                            break
                    
                    except asyncio.TimeoutError:
                        # No message received in timeout period, continue
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
                
                self.is_collecting = False
                logger.info(f"Data collection complete: {len(self.vessels)} vessels")
        
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
    
    def _parse_ais_message(self, data):
        """
        Parse AIS message from AISStream.io format
        
        Args:
            data: JSON message from WebSocket
            
        Returns:
            Dictionary with vessel data or None if invalid
        """
        
        try:
            # AISStream.io message structure
            if 'Message' not in data or 'MetaData' not in data:
                return None
            
            message = data['Message']
            metadata = data['MetaData']
            
            # Only process PositionReport messages
            if message.get('MessageType') != 'PositionReport':
                return None
            
            # Extract position report data
            position = message.get('PositionReport', {})
            
            # Get vessel metadata
            ship_static = metadata.get('ShipStaticData', {})
            
            # Build vessel dictionary
            vessel = {
                'mmsi': metadata.get('MMSI'),
                'latitude': position.get('Latitude'),
                'longitude': position.get('Longitude'),
                'speed_knots': position.get('Sog', 0),  # Speed Over Ground
                'course': position.get('Cog', 0),  # Course Over Ground
                'heading': position.get('TrueHeading', 0),
                'timestamp': datetime.fromisoformat(
                    metadata.get('time_utc', datetime.utcnow().isoformat())
                ),
                'vessel_name': ship_static.get('Name', 'Unknown'),
                'vessel_type': ship_static.get('Type', 0),
                'destination': ship_static.get('Destination', ''),
                'imo': ship_static.get('ImoNumber', 0),
                'callsign': ship_static.get('CallSign', ''),
                'nav_status': position.get('NavigationalStatus', 0)
            }
            
            # Validate required fields
            if vessel['mmsi'] and vessel['latitude'] and vessel['longitude']:
                return vessel
            
            return None
        
        except Exception as e:
            logger.debug(f"Error parsing message: {e}")
            return None
    
    def filter_high_risk_vessels(self, df):
        """
        Filter to vessels that pose higher collision risk
        
        Criteria:
        - Cargo ships (TYPE 70-79)
        - Tankers (TYPE 80-89)
        - High speed vessels (> 5 knots)
        
        Args:
            df: DataFrame with vessel data
            
        Returns:
            Filtered DataFrame
        """
        
        if df.empty:
            return df
        
        # Vessel type codes (IMO standard)
        high_risk_types = list(range(70, 90))
        
        # Filter by type OR speed
        filtered = df[
            (df['vessel_type'].isin(high_risk_types)) | 
            (df['speed_knots'] > 5)
        ].copy()
        
        logger.info(f"Filtered to {len(filtered)} high-risk vessels from {len(df)} total")
        
        return filtered
    
    def get_vessel_type_name(self, type_code):
        """
        Convert vessel type code to readable name
        
        Args:
            type_code: IMO vessel type code
            
        Returns:
            String description of vessel type
        """
        
        if pd.isna(type_code):
            return 'Unknown'
        
        type_code = int(type_code)
        
        type_mapping = {
            range(20, 30): 'Wing in Ground',
            range(30, 40): 'Fishing',
            range(40, 50): 'Towing',
            range(50, 60): 'Dredging/Underwater',
            range(60, 70): 'Passenger',
            range(70, 80): 'Cargo',
            range(80, 90): 'Tanker',
            range(90, 100): 'Other'
        }
        
        for code_range, name in type_mapping.items():
            if type_code in code_range:
                return name
        
        return 'Unknown'


def load_whale_data(whale_data_path):
    """
    Load and prepare historical whale tracking data
    
    Args:
        whale_data_path: Path to whale CSV file
        
    Returns:
        DataFrame with whale positions
    """
    
    try:
        df = pd.read_csv(whale_data_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Find latitude/longitude columns
        lat_col = [c for c in df.columns if 'lat' in c.lower()][0]
        lon_col = [c for c in df.columns if 'lon' in c.lower()][0]
        
        df.rename(columns={
            lat_col: 'latitude',
            lon_col: 'longitude'
        }, inplace=True)
        
        # Clean data
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        logger.info(f"Loaded {len(df)} whale positions")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading whale data: {e}")
        return pd.DataFrame()


def create_whale_zones(whale_df, grid_size=0.5):
    """
    Create whale concentration zones from historical data
    
    Args:
        whale_df: DataFrame with whale positions
        grid_size: Grid cell size in degrees (default 0.5 ≈ 55km)
        
    Returns:
        DataFrame with whale zone centers and densities
    """
    
    if whale_df.empty:
        return pd.DataFrame()
    
    # Create grid bins
    lat_bins = pd.cut(
        whale_df['latitude'],
        bins=int((whale_df['latitude'].max() - whale_df['latitude'].min()) / grid_size) + 1
    )
    lon_bins = pd.cut(
        whale_df['longitude'],
        bins=int((whale_df['longitude'].max() - whale_df['longitude'].min()) / grid_size) + 1
    )
    
    whale_df['lat_bin'] = lat_bins
    whale_df['lon_bin'] = lon_bins
    
    # Count sightings per grid cell
    zones = whale_df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='density')
    
    # Get bin centers
    zones['latitude'] = zones['lat_bin'].apply(lambda x: x.mid if pd.notna(x) else None)
    zones['longitude'] = zones['lon_bin'].apply(lambda x: x.mid if pd.notna(x) else None)
    
    # Remove invalid entries
    zones = zones.dropna(subset=['latitude', 'longitude'])
    
    # Keep only significant zones (top 30% by density)
    threshold = zones['density'].quantile(0.7)
    zones = zones[zones['density'] >= threshold]
    
    logger.info(f"Created {len(zones)} whale zones")
    
    return zones[['latitude', 'longitude', 'density']].reset_index(drop=True)
