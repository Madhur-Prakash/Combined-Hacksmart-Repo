import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import joblib
from app.utils.config import SCALER_PATH, FEATURES_PATH

class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        self.load_preprocessors()
    
    def load_preprocessors(self):
        try:
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_columns = joblib.load(FEATURES_PATH)
        except FileNotFoundError:
            print("Warning: Preprocessor files not found. Using default configuration.")
            self.feature_columns = [
                'current_queue', 'battery_level', 'energy_demand', 'weather_temp',
                'is_weekend', 'hour_of_day', 'station_reliability', 'energy_stability'
            ]
    
    def preprocess_request(self, data: Dict[str, Any]) -> np.ndarray:
        # Extract values directly from data dict
        timestamp = pd.to_datetime(data.get('timestamp', '2024-01-01T12:00:00'))
        hour_of_day = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_weekend = timestamp.weekday() >= 5
        
        # Get values from input data
        current_queue = data.get('current_queue', 5)
        battery_level = data.get('battery_level', 50)
        energy_demand = data.get('energy_demand', 100)
        weather_temp = data.get('weather_temp', 25)
        station_reliability = data.get('station_reliability', 0.9)
        energy_stability = data.get('energy_stability', 0.9)
        station_id = data.get('station_id', 'STATION_000')
        
        # Map API features to model features
        feature_mapping = {
            # Direct mappings
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'station_reliability_score': station_reliability,
            'energy_stability_index': energy_stability,
            
            # Derived from API inputs
            'available_batteries': current_queue * 2,  # Estimate
            'total_batteries': 50,  # Default capacity
            'available_chargers': 10,  # Default
            'total_chargers': 15,  # Default
            'power_usage_kw': energy_demand,
            'power_capacity_kw': 200,  # Default capacity
            'is_peak_hour': 1 if hour_of_day in [8,9,17,18,19] else 0,
            'traffic_factor': 1.2 if is_weekend else 1.0,
            
            # Weather condition one-hot encoding (default to Clear)
            'weather_condition_Clear': 1,
            'weather_condition_Fog': 0,
            'weather_condition_Rain': 0,
            
            # Status one-hot encoding (default to OPERATIONAL)
            'status_DEGRADED': 0,
            'status_MAINTENANCE': 0,
            'status_OPERATIONAL': 1,
            
            # Station ID one-hot encoding (default to STATION_000)
            'station_id_STATION_000': 0,
            'station_id_STATION_001': 0,
            'station_id_STATION_002': 0,
            'station_id_STATION_003': 0,
            'station_id_STATION_004': 0
        }
        
        # Handle weather condition mapping
        if weather_temp < 10:
            feature_mapping['weather_condition_Fog'] = 1
            feature_mapping['weather_condition_Clear'] = 0
        elif weather_temp > 35:
            feature_mapping['weather_condition_Rain'] = 1
            feature_mapping['weather_condition_Clear'] = 0
        
        # Handle station ID mapping
        if station_id in ['STATION_000', 'STATION_001', 'STATION_002', 'STATION_003', 'STATION_004']:
            feature_mapping[f'station_id_{station_id}'] = 1
        else:
            feature_mapping['station_id_STATION_000'] = 1  # Default
        
        # Create feature array in correct order
        model_features = [
            'available_batteries', 'total_batteries', 'available_chargers', 'total_chargers',
            'power_usage_kw', 'power_capacity_kw', 'hour_of_day', 'day_of_week',
            'is_peak_hour', 'traffic_factor', 'station_reliability_score', 'energy_stability_index',
            'weather_condition_Clear', 'weather_condition_Fog', 'weather_condition_Rain',
            'status_DEGRADED', 'status_MAINTENANCE', 'status_OPERATIONAL',
            'station_id_STATION_000', 'station_id_STATION_001', 'station_id_STATION_002',
            'station_id_STATION_003', 'station_id_STATION_004'
        ]
        
        # Build feature vector
        feature_vector = [float(feature_mapping[feature]) for feature in model_features]
        
        return np.array([feature_vector])