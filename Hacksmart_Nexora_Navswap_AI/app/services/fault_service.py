import joblib
import numpy as np
from typing import Tuple
from app.utils.config import FAULT_MODEL_PATH
from app.utils.preprocessing import DataPreprocessor

class FaultPredictionService:
    def __init__(self):
        self.fault_model = None
        self.preprocessor = DataPreprocessor()
        self.load_model()
    
    def load_model(self):
        try:
            self.fault_model = joblib.load(FAULT_MODEL_PATH)
        except FileNotFoundError as e:
            print(f"Warning: Fault model not found: {e}")
    
    def preprocess_for_fault(self, data: dict) -> np.ndarray:
        """Fault model needs 25 features instead of 23"""
        # Get base 23 features
        base_features = self.preprocessor.preprocess_request(data)[0]
        
        # Add 2 more features for fault model (25 total)
        additional_features = [
            data.get('current_queue', 5) / 10.0,  # Normalized queue factor
            1.0 - data.get('station_reliability', 0.9)  # Unreliability factor
        ]
        
        # Combine to get 25 features
        fault_features = np.concatenate([base_features, additional_features])
        return fault_features.reshape(1, -1)
    
    async def predict_fault(self, data: dict) -> Tuple[str, float]:
        if not self.fault_model:
            # Fallback prediction
            return "LOW", 0.1
        
        # Preprocess data for fault model (25 features)
        features = self.preprocess_for_fault(data)
        
        # Make prediction
        fault_prob = self.fault_model.predict_proba(features)[0][1]  # Probability of fault
        
        # Determine risk level
        if fault_prob < 0.3:
            risk_level = "LOW"
        elif fault_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return risk_level, float(fault_prob)
    
    async def enhance_station_data(self, stations_data: list) -> list:
        """Enhance multiple stations with fault predictions"""
        enhanced_stations = []
        
        for station in stations_data:
            try:
                risk_level, fault_prob = await self.predict_fault(station)
                station["fault_probability"] = fault_prob
                station["fault_risk"] = risk_level
                enhanced_stations.append(station)
            except Exception as e:
                # Use fallback values if prediction fails
                station["fault_probability"] = 0.1
                station["fault_risk"] = "LOW"
                enhanced_stations.append(station)
        
        return enhanced_stations