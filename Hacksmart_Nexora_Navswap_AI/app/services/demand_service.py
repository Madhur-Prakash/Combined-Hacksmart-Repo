import joblib
import numpy as np
import logging
from typing import Dict, Any
from app.utils.config import MODEL_DIR
from app.utils.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class DemandService:
    def __init__(self):
        self.customer_arrival_model = None
        self.battery_demand_model = None
        self.preprocessor = DataPreprocessor()
        self.load_models()
    
    def load_models(self):
        try:
            self.customer_arrival_model = joblib.load(MODEL_DIR / "customer_arrival_model.pkl")
            logger.info("Customer arrival model loaded successfully")
        except Exception as e:
            logger.warning(f"Customer arrival model failed to load: {e}")
            self.customer_arrival_model = None
        
        try:
            self.battery_demand_model = joblib.load(MODEL_DIR / "battery_demand_model.pkl")
            logger.info("Battery demand model loaded successfully")
        except Exception as e:
            logger.warning(f"Battery demand model failed to load: {e}")
            self.battery_demand_model = None
    
    async def predict_customer_arrival(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict customer arrival patterns"""
        try:
            if not self.customer_arrival_model:
                return {
                    "expected_arrivals": 8,
                    "peak_time": "14:00",
                    "arrival_rate": 0.5,
                    "confidence": 0.8
                }
            
            features = self.preprocessor.preprocess_request(data)
            prediction = self.customer_arrival_model.predict(features)[0]
            
            # Calculate arrival metrics
            expected_arrivals = max(1, int(prediction * 15))
            arrival_rate = prediction  # Customers per minute
            
            # Determine peak time based on hour of day
            current_hour = data.get('hour_of_day', 12)
            peak_hour = current_hour + 1 if prediction > 0.7 else current_hour
            peak_time = f"{peak_hour:02d}:00"
            
            return {
                "expected_arrivals": expected_arrivals,
                "peak_time": peak_time,
                "arrival_rate": float(arrival_rate),
                "confidence": min(0.95, prediction + 0.2),
                "arrival_pattern": "SURGE" if prediction > 0.8 else "STEADY"
            }
        
        except Exception as e:
            logger.error(f"Customer arrival prediction failed: {e}")
            return {"expected_arrivals": 5, "peak_time": "12:00", "arrival_rate": 0.3}
    
    async def predict_battery_demand(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict battery demand patterns"""
        try:
            if not self.battery_demand_model:
                return {
                    "demand_forecast": 12,
                    "demand_type": "STANDARD",
                    "peak_demand_hour": 15,
                    "supply_adequacy": "SUFFICIENT"
                }
            
            features = self.preprocessor.preprocess_request(data)
            prediction = self.battery_demand_model.predict(features)[0]
            
            # Calculate demand metrics
            demand_forecast = max(5, int(prediction * 20))
            demand_type = "HIGH" if prediction > 0.8 else "STANDARD" if prediction > 0.4 else "LOW"
            
            # Determine supply adequacy
            current_batteries = data.get('battery_level', 50)
            supply_ratio = current_batteries / demand_forecast
            
            if supply_ratio > 1.5:
                supply_adequacy = "EXCESS"
            elif supply_ratio > 1.0:
                supply_adequacy = "SUFFICIENT"
            elif supply_ratio > 0.7:
                supply_adequacy = "TIGHT"
            else:
                supply_adequacy = "CRITICAL"
            
            # Peak demand hour
            current_hour = data.get('hour_of_day', 12)
            peak_demand_hour = current_hour + 2 if prediction > 0.6 else current_hour + 1
            
            return {
                "demand_forecast": demand_forecast,
                "demand_type": demand_type,
                "peak_demand_hour": peak_demand_hour,
                "supply_adequacy": supply_adequacy,
                "demand_score": float(prediction),
                "recommended_stock": demand_forecast + 5  # Buffer stock
            }
        
        except Exception as e:
            logger.error(f"Battery demand prediction failed: {e}")
            return {"demand_forecast": 10, "demand_type": "STANDARD", "supply_adequacy": "SUFFICIENT"}