"""
Collision Risk Calculator
Calculates risk of ship-whale collisions
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import logging

logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth (in kilometers)
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(min(1, a)))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r


def calculate_vessel_whale_distance(vessel_row, whale_zones_df):
    """
    Calculate minimum distance from vessel to any whale zone
    
    Args:
        vessel_row: Series with vessel data (lat, lon)
        whale_zones_df: DataFrame with whale zone positions
        
    Returns:
        Tuple of (min_distance, closest_whale_lat, closest_whale_lon)
    """
    
    if whale_zones_df.empty:
        return None, None, None
    
    distances = whale_zones_df.apply(
        lambda whale: haversine_distance(
            vessel_row['latitude'],
            vessel_row['longitude'],
            whale['latitude'],
            whale['longitude']
        ),
        axis=1
    )
    
    min_idx = distances.idxmin()
    min_distance = distances[min_idx]
    
    closest_whale = whale_zones_df.loc[min_idx]
    
    return min_distance, closest_whale['latitude'], closest_whale['longitude']


def categorize_risk(distance_km, speed_knots, thresholds):
    """
    Categorize risk level based on distance and speed
    
    Args:
        distance_km: Distance to nearest whale zone
        speed_knots: Vessel speed
        thresholds: Dict with risk distance thresholds
        
    Returns:
        Risk level string
    """
    
    if distance_km <= thresholds['critical_distance_km']:
        return 'CRITICAL'
    elif distance_km <= thresholds['high_distance_km']:
        return 'HIGH'
    elif distance_km <= thresholds['medium_distance_km']:
        # Escalate to HIGH if moving fast
        if speed_knots > thresholds['dangerous_speed_knots']:
            return 'HIGH'
        return 'MEDIUM'
    elif distance_km <= thresholds['low_distance_km']:
        return 'LOW'
    else:
        return 'SAFE'


def calculate_risk_score(distance_km, speed_knots, vessel_type, thresholds):
    """
    Calculate numerical risk score (0-100)
    
    Factors:
    - Distance (closer = higher risk)
    - Speed (faster = higher risk)
    - Vessel type (cargo/tanker = higher risk)
    
    Args:
        distance_km: Distance to whale zone
        speed_knots: Vessel speed
        vessel_type: IMO vessel type code
        thresholds: Risk threshold configuration
        
    Returns:
        Risk score (0-100)
    """
    
    # Distance component (0-50 points)
    max_distance = thresholds['low_distance_km']
    distance_score = max(0, (1 - distance_km / max_distance)) * 50
    
    # Speed component (0-30 points)
    critical_speed = thresholds['dangerous_speed_knots']
    speed_score = min(speed_knots / critical_speed, 1) * 30
    
    # Vessel type component (0-20 points)
    # Cargo (70-79) and Tanker (80-89) get full points
    if 70 <= vessel_type < 90:
        type_score = 20
    else:
        type_score = 10
    
    total_score = distance_score + speed_score + type_score
    
    return min(round(total_score, 1), 100)


def assess_collision_risks(vessels_df, whale_zones_df, config):
    """
    Assess collision risk for all vessels
    
    Args:
        vessels_df: DataFrame with vessel positions
        whale_zones_df: DataFrame with whale zones
        config: Configuration dictionary
        
    Returns:
        DataFrame with risk assessments
    """
    
    if vessels_df.empty or whale_zones_df.empty:
        logger.warning("No vessels or whale zones to assess")
        return pd.DataFrame()
    
    thresholds = config['risk_thresholds']
    risk_events = []
    
    for idx, vessel in vessels_df.iterrows():
        
        # Calculate distance to nearest whale zone
        distance, whale_lat, whale_lon = calculate_vessel_whale_distance(
            vessel, whale_zones_df
        )
        
        if distance is None:
            continue
        
        # Only record if within monitoring range
        if distance > thresholds['low_distance_km']:
            continue
        
        # Categorize risk
        risk_level = categorize_risk(
            distance,
            vessel.get('speed_knots', 0),
            thresholds
        )
        
        # Calculate risk score
        risk_score = calculate_risk_score(
            distance,
            vessel.get('speed_knots', 0),
            vessel.get('vessel_type', 0),
            thresholds
        )
        
        # Create risk event
        risk_events.append({
            'mmsi': vessel['mmsi'],
            'vessel_name': vessel.get('vessel_name', 'Unknown'),
            'vessel_type': vessel.get('vessel_type', 0),
            'vessel_lat': vessel['latitude'],
            'vessel_lon': vessel['longitude'],
            'vessel_speed': vessel.get('speed_knots', 0),
            'vessel_course': vessel.get('course', 0),
            'whale_lat': whale_lat,
            'whale_lon': whale_lon,
            'distance_km': round(distance, 2),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'timestamp': vessel.get('timestamp', pd.Timestamp.now())
        })
    
    logger.info(f"Assessed {len(vessels_df)} vessels, found {len(risk_events)} at-risk")
    
    return pd.DataFrame(risk_events)


def get_risk_statistics(risk_df):
    """
    Calculate statistics from risk assessments
    
    Args:
        risk_df: DataFrame with risk events
        
    Returns:
        Dictionary with statistics
    """
    
    if risk_df.empty:
        return {
            'total_risks': 0,
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'avg_distance': 0,
            'min_distance': 0
        }
    
    return {
        'total_risks': len(risk_df),
        'critical': len(risk_df[risk_df['risk_level'] == 'CRITICAL']),
        'high': len(risk_df[risk_df['risk_level'] == 'HIGH']),
        'medium': len(risk_df[risk_df['risk_level'] == 'MEDIUM']),
        'low': len(risk_df[risk_df['risk_level'] == 'LOW']),
        'avg_distance': round(risk_df['distance_km'].mean(), 2),
        'min_distance': round(risk_df['distance_km'].min(), 2)
    }


def get_risk_color(risk_level):
    """
    Get color code for risk level
    
    Args:
        risk_level: Risk level string
        
    Returns:
        Color string
    """
    
    color_map = {
        'CRITICAL': 'red',
        'HIGH': 'orange',
        'MEDIUM': 'yellow',
        'LOW': 'lightgreen',
        'SAFE': 'green'
    }
    
    return color_map.get(risk_level, 'gray')
