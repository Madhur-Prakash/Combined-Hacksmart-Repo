import google.generativeai as genai
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LLMExplainabilityService:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")
    
    async def generate_final_explanation(
        self,
        recommendation: Dict[str, Any],
        ai_predictions: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        try:
            station_name = recommendation.get('station_name', 'Selected Station')
            score = recommendation.get('score', 0.0)
            
            prompt = f"""
            You are NavSwap's AI assistant explaining EV battery swap station recommendations.
            
            RECOMMENDATION:
            - Station: {station_name}
            - Recommendation Score: {score:.2f}/1.0
            
            AI PREDICTIONS:
            - Queue Length: {ai_predictions.get('predicted_queue', 0):.1f} vehicles
            - Wait Time: {ai_predictions.get('predicted_wait', 0):.1f} minutes
            - Fault Probability: {ai_predictions.get('fault_probability', 0):.2%}
            - System Action: {ai_predictions.get('system_action', 'NORMAL')}
            
            USER CONTEXT:
            - Battery Level: {user_context.get('battery_level', 50)}%
            - Urgency: {user_context.get('urgency', 'medium')}
            - Distance: {user_context.get('distance', 'nearby')}
            
            Provide a friendly, informative explanation (3-4 sentences) of why this station is recommended based on all AI model predictions. Focus on practical benefits for the user.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
        
        except Exception as e:
            logger.warning(f"Gemini API failed: {e}")
            return self._generate_fallback_explanation(recommendation, ai_predictions, user_context)
    
    def _generate_fallback_explanation(
        self,
        recommendation: Dict[str, Any],
        ai_predictions: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        station_name = recommendation.get('station_name', 'this station')
        queue = ai_predictions.get('predicted_queue', 0)
        wait_time = ai_predictions.get('predicted_wait', 0)
        action = ai_predictions.get('system_action', 'NORMAL')
        
        return f"NavSwap AI recommends {station_name} with {queue:.0f} vehicles in queue and {wait_time:.1f} minute wait time. System status is {action}, ensuring reliable service for your battery swap needs."