import joblib
import numpy as np
from typing import Dict, List, Any
from app.utils.config import MODEL_DIR

class StationRecommenderService:
    def __init__(self):
        self.recommender_model = None
        self.load_model()
    
    def load_model(self):
        try:
            model_path = MODEL_DIR / "station_recommender.pkl"
            self.recommender_model = joblib.load(model_path)
        except FileNotFoundError as e:
            print(f"Warning: Station recommender model not found: {e}")
    
    def apply_scoring_rules(self, stations_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply scoring rules before recommendation"""
        scored_stations = []
        
        for station in stations_data:
            # Start with base score
            score = station.get('base_score', 0.8)
            
            # Reduce score if fault probability > 0.6
            fault_prob = station.get('fault_probability', 0.0)
            if fault_prob > 0.6:
                score *= 0.5  # Reduce by 50%
            
            # Reduce score if predicted queue is high
            predicted_queue = station.get('predicted_queue', 0)
            if predicted_queue > 8:
                score *= 0.7  # Reduce by 30%
            elif predicted_queue > 5:
                score *= 0.85  # Reduce by 15%
            
            # Boost score if system action is NORMAL
            system_action = station.get('system_action', 'NORMAL')
            if system_action == 'NORMAL':
                score *= 1.2  # Boost by 20%
            elif system_action == 'MAINTENANCE_ALERT':
                score *= 0.3  # Heavily penalize
            
            station['recommendation_score'] = min(score, 1.0)  # Cap at 1.0
            scored_stations.append(station)
        
        return scored_stations
    
    async def get_recommendation(self, stations_data: List[Dict[str, Any]], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get station recommendation using the trained model"""
        if not self.recommender_model:
            # Fallback: return highest scored station
            scored_stations = self.apply_scoring_rules(stations_data)
            sorted_stations = sorted(scored_stations, key=lambda x: x.get('recommendation_score', 0), reverse=True)
            
            return {
                'recommended_station': sorted_stations[0] if sorted_stations else {},
                'alternatives': sorted_stations[1:3] if len(sorted_stations) > 1 else [],
                'confidence': 0.7
            }
        
        try:
            # Apply scoring rules first
            scored_stations = self.apply_scoring_rules(stations_data)
            
            # Use the trained recommender model
            recommendation_result = self.recommender_model.get_recommendation(scored_stations, user_context)
            
            return recommendation_result
        
        except Exception as e:
            print(f"Recommender model failed: {e}")
            # Fallback to rule-based recommendation
            scored_stations = self.apply_scoring_rules(stations_data)
            sorted_stations = sorted(scored_stations, key=lambda x: x.get('recommendation_score', 0), reverse=True)
            
            return {
                'recommended_station': sorted_stations[0] if sorted_stations else {},
                'alternatives': sorted_stations[1:3] if len(sorted_stations) > 1 else [],
                'confidence': 0.6
            }