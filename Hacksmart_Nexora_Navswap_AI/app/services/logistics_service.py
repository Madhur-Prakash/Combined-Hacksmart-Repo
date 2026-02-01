import joblib
import numpy as np
import logging
from typing import Dict, Any, List
from app.utils.config import MODEL_DIR
from app.utils.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)

class LogisticsService:
    def __init__(self):
        self.battery_rebalance_model = None
        self.stock_order_model = None
        self.tieup_storage_model = None
        self.preprocessor = DataPreprocessor()
        self.load_models()
    
    def load_models(self):
        try:
            self.battery_rebalance_model = joblib.load(MODEL_DIR / "battery_rebalance_model.pkl")
            logger.info("Battery rebalance model loaded successfully")
        except Exception as e:
            logger.warning(f"Battery rebalance model failed to load: {e}")
            self.battery_rebalance_model = None
        
        try:
            self.stock_order_model = joblib.load(MODEL_DIR / "stock_order_model.pkl")
            logger.info("Stock order model loaded successfully")
        except Exception as e:
            logger.warning(f"Stock order model failed to load: {e}")
            self.stock_order_model = None
        
        try:
            self.tieup_storage_model = joblib.load(MODEL_DIR / "tieup_storage_model.pkl")
            logger.info("Tieup storage model loaded successfully")
        except Exception as e:
            logger.warning(f"Tieup storage model failed to load: {e}")
            self.tieup_storage_model = None
    
    async def predict_battery_rebalance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict battery rebalancing requirements"""
        try:
            if not self.battery_rebalance_model:
                # More realistic fallback based on input data
                current_queue = data.get('current_queue', 5)
                battery_level = data.get('battery_level', 50)
                
                # Calculate realistic rebalancing needs
                batteries_needed = max(2, int(current_queue * 1.5))
                rebalance_score = min(0.8, (current_queue / 10) + (1 - battery_level / 100))
                
                return {
                    "rebalance_needed": current_queue > 3 or battery_level < 60,
                    "batteries_to_move": batteries_needed,
                    "priority": "HIGH" if current_queue > 6 else "MEDIUM",
                    "estimated_time": batteries_needed * 6,
                    "rebalance_score": rebalance_score
                }
            
            features = self.preprocessor.preprocess_request(data)
            prediction = self.battery_rebalance_model.predict(features)[0]
            
            # Determine rebalancing plan
            batteries_to_move = max(1, int(prediction * 10))
            priority = "HIGH" if prediction > 0.7 else "MEDIUM" if prediction > 0.4 else "LOW"
            
            return {
                "rebalance_needed": prediction > 0.3,
                "batteries_to_move": batteries_to_move,
                "priority": priority,
                "estimated_time": batteries_to_move * 6,
                "rebalance_score": float(prediction)
            }
        
        except Exception as e:
            logger.error(f"Battery rebalance prediction failed: {e}")
            return {"rebalance_needed": True, "batteries_to_move": 3, "priority": "MEDIUM"}
    
    async def predict_stock_orders(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict stock ordering requirements"""
        try:
            if not self.stock_order_model:
                # Realistic fallback based on inventory levels
                current_queue = data.get('current_queue', 5)
                battery_level = data.get('battery_level', 50)
                hour_of_day = data.get('hour_of_day', 12)
                
                # Calculate realistic order needs
                daily_demand = current_queue * 3  # Estimate daily demand
                order_quantity = max(8, int(daily_demand * 0.6))
                
                # Determine urgency based on current situation
                if battery_level < 40 or current_queue > 6:
                    urgency = "HIGH"
                    delivery_window = "6h"
                elif battery_level < 70 or current_queue > 3:
                    urgency = "MEDIUM" 
                    delivery_window = "12h"
                else:
                    urgency = "LOW"
                    delivery_window = "24h"
                
                return {
                    "order_needed": battery_level < 80 or current_queue > 2,
                    "battery_quantity": order_quantity,
                    "urgency": urgency,
                    "delivery_window": delivery_window,
                    "order_score": min(0.9, (current_queue / 8) + (1 - battery_level / 100))
                }
            
            features = self.preprocessor.preprocess_request(data)
            prediction = self.stock_order_model.predict(features)[0]
            
            # Determine order requirements
            battery_quantity = max(5, int(prediction * 20))
            urgency = "HIGH" if prediction > 0.8 else "MEDIUM" if prediction > 0.5 else "LOW"
            delivery_window = "12h" if urgency == "HIGH" else "24h" if urgency == "MEDIUM" else "48h"
            
            return {
                "order_needed": prediction > 0.4,
                "battery_quantity": battery_quantity,
                "urgency": urgency,
                "delivery_window": delivery_window,
                "order_score": float(prediction)
            }
        
        except Exception as e:
            logger.error(f"Stock order prediction failed: {e}")
            return {"order_needed": True, "battery_quantity": 12, "urgency": "MEDIUM"}
    
    async def predict_tieup_storage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict partner storage requirements"""
        try:
            if not self.tieup_storage_model:
                # Realistic fallback based on operational conditions
                current_queue = data.get('current_queue', 5)
                battery_level = data.get('battery_level', 50)
                system_action = data.get('system_action', 'NORMAL')
                
                # Calculate realistic storage needs
                if system_action == 'MAINTENANCE_ALERT' or current_queue > 8:
                    storage_needed = True
                    partner_stations = ["PARTNER_A", "PARTNER_B"]
                    storage_duration = "12h"
                    storage_capacity = 8
                elif current_queue > 5 or battery_level < 40:
                    storage_needed = True
                    partner_stations = ["PARTNER_A"]
                    storage_duration = "6h"
                    storage_capacity = 5
                else:
                    storage_needed = False
                    partner_stations = []
                    storage_duration = "3h"
                    storage_capacity = 0
                
                return {
                    "storage_needed": storage_needed,
                    "partner_stations": partner_stations,
                    "storage_duration": storage_duration,
                    "storage_capacity": storage_capacity,
                    "storage_score": min(0.8, current_queue / 10)
                }
            
            features = self.preprocessor.preprocess_request(data)
            prediction = self.tieup_storage_model.predict(features)[0]
            
            # Determine storage plan
            partner_stations = ["PARTNER_A", "PARTNER_B"] if prediction > 0.6 else []
            storage_duration = "12h" if prediction > 0.8 else "6h" if prediction > 0.5 else "3h"
            
            return {
                "storage_needed": prediction > 0.4,
                "partner_stations": partner_stations,
                "storage_duration": storage_duration,
                "storage_capacity": int(prediction * 15),
                "storage_score": float(prediction)
            }
        
        except Exception as e:
            logger.error(f"Tieup storage prediction failed: {e}")
            return {"storage_needed": True, "partner_stations": ["PARTNER_A"], "storage_capacity": 5}