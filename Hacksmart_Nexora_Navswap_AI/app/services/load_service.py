import joblib
import numpy as np
from typing import Tuple, Dict, Any
from app.utils.config import QUEUE_MODEL_PATH, WAIT_MODEL_PATH
from app.utils.preprocessing import DataPreprocessor

class LoadPredictionService:
    def __init__(self):
        self.queue_model = None
        self.wait_model = None
        self.preprocessor = DataPreprocessor()
        self.load_models()
    
    def load_models(self):
        try:
            self.queue_model = joblib.load(QUEUE_MODEL_PATH)
            self.wait_model = joblib.load(WAIT_MODEL_PATH)
        except FileNotFoundError as e:
            print(f"Warning: Model files not found: {e}")
    
    async def predict_load(self, data: dict) -> Tuple[float, float]:
        if not self.queue_model or not self.wait_model:
            # Fallback predictions
            return 5.0, 15.0
        
        # Preprocess data
        features = self.preprocessor.preprocess_request(data)
        
        # Make predictions
        queue_pred = self.queue_model.predict(features)[0]
        wait_pred = self.wait_model.predict(features)[0]
        
        return float(queue_pred), float(wait_pred)
    
    async def enhance_station_data(self, stations_data: list) -> list:
        """Enhance multiple stations with load predictions"""
        enhanced_stations = []
        
        for station in stations_data:
            try:
                queue_pred, wait_pred = await self.predict_load(station)
                station["predicted_queue"] = queue_pred
                station["predicted_wait"] = wait_pred
                enhanced_stations.append(station)
            except Exception as e:
                # Use fallback values if prediction fails
                station["predicted_queue"] = 5.0
                station["predicted_wait"] = 15.0
                enhanced_stations.append(station)
        
        return enhanced_stations