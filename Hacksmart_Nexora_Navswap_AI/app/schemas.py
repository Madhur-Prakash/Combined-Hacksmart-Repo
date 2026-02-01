from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    timestamp: datetime
    station_id: str
    current_queue: int
    battery_level: float
    energy_demand: float
    weather_temp: float
    is_weekend: bool
    hour_of_day: int
    station_reliability: float
    energy_stability: float

class LoadPredictionResponse(BaseModel):
    predicted_queue_length: float
    predicted_wait_time: float

class FaultPredictionResponse(BaseModel):
    fault_risk: str
    fault_probability: float

class ActionPredictionResponse(BaseModel):
    system_action: str
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