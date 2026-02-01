from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.schemas import (
    PredictionRequest, LoadPredictionResponse, FaultPredictionResponse,
    ActionPredictionResponse, ExplanationRequest, ExplanationResponse
)
from app.services.load_service import LoadPredictionService
from app.services.fault_service import FaultPredictionService
from app.services.action_service import ActionPredictionService
from app.services.explain_service import ExplanationService
from app.services.station_recommender_service import StationRecommenderService
from app.services.llm_explainability_service import LLMExplainabilityService
from app.services.traffic_service import TrafficService
from app.services.logistics_service import LogisticsService
from app.services.staff_service import StaffService
from app.services.demand_service import DemandService
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
load_service = LoadPredictionService()
fault_service = FaultPredictionService()
action_service = ActionPredictionService()
explain_service = ExplanationService()
recommender_service = StationRecommenderService()
llm_service = LLMExplainabilityService()
traffic_service = TrafficService()
logistics_service = LogisticsService()
staff_service = StaffService()
demand_service = DemandService()

# New schemas for unified recommendation
class StationData(BaseModel):
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    current_queue: int
    battery_level: float
    energy_demand: float
    weather_temp: float
    is_weekend: bool
    hour_of_day: int
    station_reliability: float
    energy_stability: float
    base_score: float = 0.8

class UserContext(BaseModel):
    battery_level: int
    urgency: str = "medium"
    distance: str = "nearby"
    max_wait_time: int = 30

class SmartRecommendationRequest(BaseModel):
    user_context: UserContext
    stations_data: List[StationData]

class SmartRecommendationResponse(BaseModel):
    recommended_station: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    ai_predictions: Dict[str, Any]
    explanation: str
    confidence: float

# New schemas for operations optimization
class OperationsRequest(BaseModel):
    station_metrics: Dict[str, Any]
    traffic_metrics: Dict[str, Any]
    staff_availability: Dict[str, Any]
    inventory_levels: Dict[str, Any]
    customer_context: Dict[str, Any]

class OperationsResponse(BaseModel):
    station_recommendation: Dict[str, Any]
    traffic_prediction: Dict[str, Any]
    battery_transport_plan: Dict[str, Any]
    staff_diversion_plan: Dict[str, Any]
    inventory_order_plan: Dict[str, Any]
    partner_storage_plan: Dict[str, Any]
    ai_explanation: str
    confidence_score: float

@router.post("/smart-recommend", response_model=SmartRecommendationResponse)
async def smart_recommend(request: SmartRecommendationRequest):
    """Unified AI pipeline for intelligent station recommendation"""
    try:
        # Convert to list of dicts
        stations_data = [station.dict() for station in request.stations_data]
        user_context = request.user_context.dict()
        
        # Step 1: Run all 4 prediction models on each station
        enhanced_stations = await load_service.enhance_station_data(stations_data)
        enhanced_stations = await fault_service.enhance_station_data(enhanced_stations)
        enhanced_stations = await action_service.enhance_station_data(enhanced_stations)
        
        # Step 2: Get recommendation using enhanced data
        recommendation_result = await recommender_service.get_recommendation(
            enhanced_stations, user_context
        )
        
        # Step 3: Prepare AI predictions summary
        recommended_station = recommendation_result['recommended_station']
        ai_predictions = {
            "predicted_queue": recommended_station.get('predicted_queue', 0),
            "predicted_wait": recommended_station.get('predicted_wait', 0),
            "fault_probability": recommended_station.get('fault_probability', 0),
            "system_action": recommended_station.get('system_action', 'NORMAL')
        }
        
        # Step 4: Generate final explanation using LLM
        explanation = await llm_service.generate_final_explanation(
            recommended_station, ai_predictions, user_context
        )
        
        return SmartRecommendationResponse(
            recommended_station=recommended_station,
            alternatives=recommendation_result.get('alternatives', []),
            ai_predictions=ai_predictions,
            explanation=explanation,
            confidence=recommendation_result.get('confidence', 0.8)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart recommendation failed: {str(e)}")

@router.post("/smart-operations", response_model=OperationsResponse)
async def smart_operations(request: OperationsRequest):
    """Comprehensive operations optimization using all AI models"""
    try:
        logger.info("Starting smart operations analysis")
        
        # Combine all input data
        combined_data = {
            **request.station_metrics,
            **request.traffic_metrics,
            **request.customer_context
        }
        
        # Step 1: Traffic Analysis
        logger.info("Analyzing traffic patterns")
        micro_traffic = await traffic_service.predict_micro_traffic(combined_data)
        future_traffic = await traffic_service.predict_future_traffic(combined_data)
        traffic_prediction = {**micro_traffic, **future_traffic}
        
        # Step 2: Core Predictions (Load, Fault, Action)
        logger.info("Running core prediction models")
        queue_pred, wait_pred = await load_service.predict_load(combined_data)
        fault_risk, fault_prob = await fault_service.predict_fault(combined_data)
        action, action_probs = await action_service.predict_action(combined_data)
        
        # Step 3: Logistics Planning (pass action result)
        logger.info("Planning logistics operations")
        combined_data_with_action = {**combined_data, "system_action": action}
        battery_transport_plan = await logistics_service.predict_battery_rebalance(combined_data_with_action)
        inventory_order_plan = await logistics_service.predict_stock_orders(combined_data_with_action)
        partner_storage_plan = await logistics_service.predict_tieup_storage(combined_data_with_action)
        
        # Step 4: Staff Planning (pass action result)
        logger.info("Planning staff operations")
        combined_data_with_action = {**combined_data, "system_action": action}
        staff_diversion_plan = await staff_service.predict_staff_diversion(combined_data_with_action)
        
        # Step 5: Demand Analysis
        logger.info("Analyzing demand patterns")
        customer_arrivals = await demand_service.predict_customer_arrival(combined_data)
        battery_demand = await demand_service.predict_battery_demand(combined_data)
        
        # Step 6: Station Recommendation (enhanced with all predictions)
        enhanced_station_data = [{
            **combined_data,
            "predicted_queue": queue_pred,
            "predicted_wait": wait_pred,
            "fault_probability": fault_prob,
            "system_action": action,
            "station_name": combined_data.get('station_id', 'Current Station')
        }]
        
        recommendation_result = await recommender_service.get_recommendation(
            enhanced_station_data, request.customer_context
        )
        
        station_recommendation = recommendation_result['recommended_station']
        
        # Step 7: Comprehensive AI Explanation
        logger.info("Generating comprehensive explanation")
        operations_data = {
            "station_recommendation": station_recommendation,
            "traffic_prediction": traffic_prediction,
            "battery_transport_plan": battery_transport_plan,
            "staff_diversion_plan": staff_diversion_plan,
            "inventory_order_plan": inventory_order_plan,
            "partner_storage_plan": partner_storage_plan
        }
        
        ai_explanation = await explain_service.explain_operations(operations_data)
        
        # Calculate overall confidence score
        confidence_scores = [
            recommendation_result.get('confidence', 0.8),
            traffic_prediction.get('micro_traffic_score', 0.7),
            battery_transport_plan.get('rebalance_score', 0.7),
            staff_diversion_plan.get('diversion_score', 0.7)
        ]
        overall_confidence = sum(confidence_scores) / len(confidence_scores)
        
        logger.info("Smart operations analysis completed successfully")
        
        return OperationsResponse(
            station_recommendation=station_recommendation,
            traffic_prediction=traffic_prediction,
            battery_transport_plan=battery_transport_plan,
            staff_diversion_plan=staff_diversion_plan,
            inventory_order_plan=inventory_order_plan,
            partner_storage_plan=partner_storage_plan,
            ai_explanation=ai_explanation,
            confidence_score=overall_confidence
        )
        
    except Exception as e:
        logger.error(f"Smart operations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Smart operations failed: {str(e)}")

@router.post("/predict-load", response_model=LoadPredictionResponse)
async def predict_load(request: PredictionRequest):
    try:
        data = request.dict()
        queue_pred, wait_pred = await load_service.predict_load(data)
        
        return LoadPredictionResponse(
            predicted_queue_length=queue_pred,
            predicted_wait_time=wait_pred
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load prediction failed: {str(e)}")

@router.post("/predict-fault", response_model=FaultPredictionResponse)
async def predict_fault(request: PredictionRequest):
    try:
        data = request.dict()
        risk_level, fault_prob = await fault_service.predict_fault(data)
        
        return FaultPredictionResponse(
            fault_risk=risk_level,
            fault_probability=fault_prob
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fault prediction failed: {str(e)}")

@router.post("/predict-action", response_model=ActionPredictionResponse)
async def predict_action(request: PredictionRequest):
    try:
        data = request.dict()
        action, probabilities = await action_service.predict_action(data)
        
        return ActionPredictionResponse(
            system_action=action,
            action_probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action prediction failed: {str(e)}")

@router.post("/explain-decision", response_model=ExplanationResponse)
async def explain_decision(request: ExplanationRequest):
    try:
        explanation = await explain_service.explain_decision(
            request.action,
            request.queue_prediction,
            request.wait_time,
            request.fault_probability,
            request.station_reliability,
            request.energy_stability
        )
        
        return ExplanationResponse(explanation=explanation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")