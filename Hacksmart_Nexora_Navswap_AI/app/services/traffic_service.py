import joblib
import numpy as np
import logging
from typing import Dict, Any
from app.utils.config import MODEL_DIR
from app.utils.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class TrafficService:
    def __init__(self):
        self.traffic_forecast_model = None
        self.micro_traffic_model = None
        self.preprocessor = DataPreprocessor()
        self.load_models()
    
    def load_models(self):
        try:
            self.traffic_forecast_model = joblib.load(MODEL_DIR / "traffic_forecast_model.pkl")
            logger.info("Traffic forecast model loaded successfully")
        except Exception as e:
            logger.warning(f"Traffic forecast model failed to load: {e}")
            self.traffic_forecast_model = None
        
        try:
            self.micro_traffic_model = joblib.load(MODEL_DIR / "micro_traffic_model_improved.pkl")
            logger.info("Micro traffic model loaded successfully")
        except Exception as e:
            logger.warning(f"Micro traffic model failed to load: {e}")
            self.micro_traffic_model = None
    
    async def predict_micro_traffic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict micro-level traffic patterns"""
        try:
            if not self.micro_traffic_model:
                return {"micro_traffic_score": 0.7, "congestion_level": "MEDIUM"}
            
            features = self.preprocessor.preprocess_request(data)
            prediction = self.micro_traffic_model.predict(features)[0]
            
            # Determine congestion level
            if prediction < 0.3:
                congestion_level = "LOW"
            elif prediction < 0.7:
                congestion_level = "MEDIUM"
            else:
                congestion_level = "HIGH"
            
            return {
                "micro_traffic_score": float(prediction),
                "congestion_level": congestion_level,
                "estimated_delay": prediction * 10  # Convert to minutes
            }
        
        except Exception as e:
            logger.error(f"Micro traffic prediction failed: {e}")
            return {"micro_traffic_score": 0.5, "congestion_level": "MEDIUM"}
    
    async def predict_future_traffic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future traffic patterns"""
        try:
            if not self.traffic_forecast_model:
                return {"traffic_forecast": 0.6, "peak_hours": [8, 17, 18]}
            
            features = self.preprocessor.preprocess_request(data)
            prediction = self.traffic_forecast_model.predict(features)[0]
            
            # Predict peak hours based on traffic score
            peak_hours = [8, 9, 17, 18, 19] if prediction > 0.7 else [8, 17, 18]
            
            return {
                "traffic_forecast": float(prediction),
                "peak_hours": peak_hours,
                "traffic_trend": "INCREASING" if prediction > 0.6 else "STABLE"
            }
        
        except Exception as e:
            logger.error(f"Future traffic prediction failed: {e}")
            return {"traffic_forecast": 0.5, "peak_hours": [8, 17, 18]}