from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class StationMetrics(BaseModel):
    station_id: str
    timestamp: Optional[str] = None # ISO format
    # These match the features expected by the models (based on description)
    # We might need to adjust these based on the exact columns in the scaler
    # For now, I'll include the ones mentioned in the context.
    
    # Operational Metrics
    queue_length: Optional[float] = None
    available_batteries: Optional[float] = None
    total_batteries: Optional[float] = None
    available_chargers: Optional[float] = None
    total_chargers: Optional[float] = None
    faulty_chargers: Optional[float] = None
    avg_wait_time: Optional[float] = None
    power_usage_kw: Optional[float] = None
    power_capacity_kw: Optional[float] = None
    fault_count_24h: Optional[float] = None
    
    # Contextual Factors
    traffic_factor: Optional[float] = None
    station_reliability_score: Optional[float] = None
    energy_stability_index: Optional[float] = None
    
    # Categorical (will need encoding)
    weather_condition: Optional[str] = None
    status: Optional[str] = None
    
    # Time-series features (can be derived from timestamp if not provided, 
    # but models might expect them directly)
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    is_peak_hour: Optional[int] = None

class LoadPredictionResponse(BaseModel):
    predicted_queue_length: float
    predicted_avg_wait_time: float

class FaultPredictionResponse(BaseModel):
    fault_risk: int # 0 or 1
    fault_probability: float

class ActionPredictionResponse(BaseModel):
    action_encoded: int
    action_label: str
    action_probabilities: Dict[str, float]

class ExplanationRequest(BaseModel):
    action: str
    queue_prediction: float
    wait_time: float
    fault_probability: float
    station_reliability: float
    energy_stability: float

class ExplanationResponse(BaseModel):
    explanation: str
