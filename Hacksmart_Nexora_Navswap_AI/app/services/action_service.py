import joblib
import numpy as np
from typing import Tuple, Dict
from app.utils.config import ACTION_MODEL_PATH, ENCODER_PATH
from app.utils.preprocessing import DataPreprocessor

class ActionPredictionService:
    def __init__(self):
        self.action_model = None
        self.label_encoder = None
        self.preprocessor = DataPreprocessor()
        self.load_models()
    
    def load_models(self):
        try:
            self.action_model = joblib.load(ACTION_MODEL_PATH)
            self.label_encoder = joblib.load(ENCODER_PATH)
        except FileNotFoundError as e:
            print(f"Warning: Action model files not found: {e}")
    
    async def predict_action(self, data: dict) -> Tuple[str, Dict[str, float]]:
        if not self.action_model or not self.label_encoder:
            # Fallback prediction
            return "NORMAL", {"NORMAL": 0.8, "REDIRECT": 0.15, "MAINTENANCE_ALERT": 0.05}
        
        # Preprocess data
        features = self.preprocessor.preprocess_request(data)
        
        # Make prediction
        action_probs = self.action_model.predict_proba(features)[0]
        predicted_class = self.action_model.predict(features)[0]
        
        # Decode action
        action = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # Create probability dictionary
        action_classes = self.label_encoder.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(action_classes, action_probs)}
        
        return action, prob_dict
    
    async def enhance_station_data(self, stations_data: list) -> list:
        """Enhance multiple stations with action predictions"""
        enhanced_stations = []
        
        for station in stations_data:
            try:
                action, probabilities = await self.predict_action(station)
                station["system_action"] = action
                station["action_probabilities"] = probabilities
                enhanced_stations.append(station)
            except Exception as e:
                # Use fallback values if prediction fails
                station["system_action"] = "NORMAL"
                station["action_probabilities"] = {"NORMAL": 0.8, "REDIRECT": 0.15, "MAINTENANCE_ALERT": 0.05}
                enhanced_stations.append(station)
        
        return enhanced_stations